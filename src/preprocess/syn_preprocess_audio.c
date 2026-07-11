/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_preprocess_audio.c
 * @brief SynapticOS — Audio pre-processor (MFCC)
 *
 * MFCC feature extraction: Hamming window -> FFT (DSP HAL) -> power
 * spectrum -> triangular mel filterbank -> log -> DCT-II.
 *
 * Deliberate simplifications for the embedded profile (documented,
 * not librosa-parity): non-overlapping frames, no pre-emphasis,
 * natural log, unnormalized DCT-II. Suitable for keyword-spotting
 * models trained with the same frontend.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_infer.h>
#include <synaptic/syn_process.h>
#include <synaptic/syn_mem.h>
#include <synaptic/syn_hal_dsp.h>
#include <string.h>
#include <math.h>

LOG_MODULE_REGISTER(syn_pre_audio, CONFIG_SYNAPTIC_LOG_LEVEL);

#define MFCC_MAX_FRAME_LEN 512
#define MFCC_MAX_MEL       32
#define MFCC_PI            3.14159265358979f

static float hz_to_mel(float hz)
{
	return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
	return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static int audio_mfcc(const syn_tensor_t *in, syn_tensor_t *out,
		      const void *config)
{
	const syn_mfcc_config_t *cfg = config;

	if (in == NULL || in->data == NULL || out == NULL ||
	    out->data == NULL || cfg == NULL ||
	    in->dtype != SYN_NPU_DTYPE_FLOAT32) {
		return -EINVAL;
	}
	if (cfg->sample_rate_hz == 0 || cfg->frame_len < 8 ||
	    cfg->frame_len > MFCC_MAX_FRAME_LEN ||
	    (cfg->frame_len & (cfg->frame_len - 1)) != 0 ||
	    cfg->num_mel == 0 || cfg->num_mel > MFCC_MAX_MEL ||
	    cfg->num_coeffs == 0 || cfg->num_coeffs > cfg->num_mel) {
		return -EINVAL;
	}

	size_t num_samples = in->size / sizeof(float);
	uint32_t frame_len = cfg->frame_len;
	uint32_t num_frames = num_samples / frame_len;
	uint32_t num_bins = frame_len / 2 + 1;

	if (num_frames == 0) {
		return -EINVAL;
	}

	size_t needed = (size_t)num_frames * cfg->num_coeffs * sizeof(float);

	if (out->size < needed) {
		LOG_ERR("MFCC output needs %u bytes, capacity %u",
			(unsigned)needed, (unsigned)out->size);
		return -ENOMEM;
	}

	/* Complex FFT working buffer from the scratch pool */
	float *fft_buf = syn_mem_scratch_acquire(frame_len * 2 *
						 sizeof(float));

	if (fft_buf == NULL) {
		LOG_ERR("MFCC: scratch pool too small for frame_len %u",
			frame_len);
		return -ENOMEM;
	}

	/* Mel filter edges: num_mel + 2 points spaced evenly on the mel
	 * scale between 0 Hz and Nyquist, converted to FFT bin numbers.
	 */
	float mel_max = hz_to_mel((float)cfg->sample_rate_hz / 2.0f);
	float bin_edges[MFCC_MAX_MEL + 2];

	for (uint32_t m = 0; m < (uint32_t)cfg->num_mel + 2; m++) {
		float mel = mel_max * (float)m / (float)(cfg->num_mel + 1);
		float hz = mel_to_hz(mel);

		bin_edges[m] = hz * (float)frame_len /
			       (float)cfg->sample_rate_hz;
	}

	const float *samples = in->data;
	float *coeffs = out->data;

	for (uint32_t f = 0; f < num_frames; f++) {
		const float *frame = &samples[(size_t)f * frame_len];

		/* Hamming window into the complex buffer */
		for (uint32_t i = 0; i < frame_len; i++) {
			float w = 0.54f - 0.46f *
				cosf(2.0f * MFCC_PI * i / (frame_len - 1));

			fft_buf[2 * i] = frame[i] * w;
			fft_buf[2 * i + 1] = 0.0f;
		}

		int ret = syn_hal_dsp_fft_f32(fft_buf, fft_buf, frame_len);

		if (ret != 0) {
			syn_mem_scratch_release(fft_buf);
			return ret;
		}

		/* Log-mel energies via triangular filters over the power
		 * spectrum (computed on the fly from the FFT bins).
		 */
		float log_mel[MFCC_MAX_MEL];

		for (uint32_t m = 0; m < cfg->num_mel; m++) {
			float lo = bin_edges[m];
			float mid = bin_edges[m + 1];
			float hi = bin_edges[m + 2];
			float energy = 0.0f;

			for (uint32_t k = 0; k < num_bins; k++) {
				float pos = (float)k;
				float weight;

				if (pos <= lo || pos >= hi) {
					continue;
				}
				if (pos <= mid) {
					weight = (mid > lo) ?
						(pos - lo) / (mid - lo) : 0.0f;
				} else {
					weight = (hi > mid) ?
						(hi - pos) / (hi - mid) : 0.0f;
				}

				float re = fft_buf[2 * k];
				float im = fft_buf[2 * k + 1];

				energy += weight * (re * re + im * im);
			}

			log_mel[m] = logf(energy + 1e-6f);
		}

		/* DCT-II down to num_coeffs cepstral coefficients */
		for (uint32_t j = 0; j < cfg->num_coeffs; j++) {
			float acc = 0.0f;

			for (uint32_t m = 0; m < cfg->num_mel; m++) {
				acc += log_mel[m] *
				       cosf(MFCC_PI * j * (m + 0.5f) /
					    cfg->num_mel);
			}
			coeffs[(size_t)f * cfg->num_coeffs + j] = acc;
		}
	}

	syn_mem_scratch_release(fft_buf);

	out->size = needed;
	out->dtype = SYN_NPU_DTYPE_FLOAT32;
	out->ndim = 2;
	memset(out->shape, 0, sizeof(out->shape));
	out->shape[0] = num_frames;
	out->shape[1] = cfg->num_coeffs;
	return 0;
}

syn_preprocess_fn_t syn_preprocess_audio_mfcc = audio_mfcc;
