/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_dsp_hal.c
 * @brief Unit tests for DSP HAL stub
 */

#include <zephyr/ztest.h>
#include <synaptic/syn_hal_dsp.h>
#include <math.h>
#include <string.h>

ZTEST_SUITE(syn_dsp_suite, NULL, NULL, NULL, NULL, NULL);

ZTEST(syn_dsp_suite, test_dsp_init)
{
    int ret = syn_hal_dsp_init();
    zassert_equal(ret, 0, "DSP init failed: %d", ret);
}

ZTEST(syn_dsp_suite, test_normalize_basic)
{
    /* scale = 1.0/255.0, zero_point = -128 maps [0,255] -> [-128,127] */
    uint8_t in[] = {0, 128, 255};
    int8_t out[3];
    float scale = 1.0f / 255.0f;
    int32_t zp = -128;

    int ret = syn_hal_dsp_normalize_int8(in, out, 3, scale, zp);
    zassert_equal(ret, 0, "normalize failed: %d", ret);

    /* in[0]=0: (int32_t)(0*scale) + (-128) = -128 */
    zassert_equal(out[0], -128, "out[0] wrong: %d", out[0]);
    /* in[2]=255: (int32_t)(255*scale) + (-128) = (int32_t)(1.0) + (-128) = 1-128 = -127 */
    zassert_equal(out[2], -127, "out[2] wrong: %d", out[2]);
}

ZTEST(syn_dsp_suite, test_normalize_null)
{
    uint8_t in[4];
    int8_t out[4];

    zassert_equal(syn_hal_dsp_normalize_int8(NULL, out, 4, 1.0f, 0), -EINVAL,
                  "Should reject NULL input");
    zassert_equal(syn_hal_dsp_normalize_int8(in, NULL, 4, 1.0f, 0), -EINVAL,
                  "Should reject NULL output");
    zassert_equal(syn_hal_dsp_normalize_int8(in, out, 0, 1.0f, 0), -EINVAL,
                  "Should reject zero length");
}

ZTEST(syn_dsp_suite, test_softmax_basic)
{
    float in[] = {1.0f, 2.0f, 3.0f};
    float out[3];

    int ret = syn_hal_dsp_softmax_f32(in, out, 3);
    zassert_equal(ret, 0, "softmax failed: %d", ret);

    /* Verify sum is 1.0 */
    float sum = out[0] + out[1] + out[2];
    zassert_true(fabsf(sum - 1.0f) < 0.001f,
                 "Softmax sum should be ~1.0, got %f", (double)sum);

    /* Verify ordering: out[2] > out[1] > out[0] */
    zassert_true(out[2] > out[1], "out[2] should be > out[1]");
    zassert_true(out[1] > out[0], "out[1] should be > out[0]");
}

ZTEST(syn_dsp_suite, test_softmax_null)
{
    float in[3], out[3];

    zassert_equal(syn_hal_dsp_softmax_f32(NULL, out, 3), -EINVAL, "");
    zassert_equal(syn_hal_dsp_softmax_f32(in, NULL, 3), -EINVAL, "");
    zassert_equal(syn_hal_dsp_softmax_f32(in, out, 0), -EINVAL, "");
}

ZTEST(syn_dsp_suite, test_argmax_basic)
{
    int8_t data[] = {-10, 5, 127, -128, 0};
    uint32_t idx;

    int ret = syn_hal_dsp_argmax(data, 5, &idx);
    zassert_equal(ret, 0, "argmax failed: %d", ret);
    zassert_equal(idx, 2, "Max should be at index 2, got %u", idx);
}

ZTEST(syn_dsp_suite, test_argmax_all_equal)
{
    int8_t data[] = {42, 42, 42};
    uint32_t idx;

    int ret = syn_hal_dsp_argmax(data, 3, &idx);
    zassert_equal(ret, 0, "argmax failed: %d", ret);
    /* When all equal, first index should be returned */
    zassert_equal(idx, 0, "Should return first index when all equal");
}

ZTEST(syn_dsp_suite, test_argmax_negative)
{
    int8_t data[] = {-100, -50, -10, -80};
    uint32_t idx;

    int ret = syn_hal_dsp_argmax(data, 4, &idx);
    zassert_equal(ret, 0, "argmax failed");
    zassert_equal(idx, 2, "Max should be at index 2 (value -10)");
}

ZTEST(syn_dsp_suite, test_argmax_null)
{
    int8_t data[4];
    uint32_t idx;

    zassert_equal(syn_hal_dsp_argmax(NULL, 4, &idx), -EINVAL, "");
    zassert_equal(syn_hal_dsp_argmax(data, 4, NULL), -EINVAL, "");
    zassert_equal(syn_hal_dsp_argmax(data, 0, &idx), -EINVAL, "");
}

ZTEST(syn_dsp_suite, test_fft_supported)
{
    /* Implemented in Phase 2: complex input needs 2*len floats */
    float in[8] = {0}, out[8] = {0};

    in[0] = 1.0f;
    zassert_equal(syn_hal_dsp_fft_f32(in, out, 4), 0,
                  "FFT should succeed");
}

ZTEST(syn_dsp_suite, test_mat_mult_supported)
{
    /* Implemented in Phase 2: b is a cols-length vector */
    int16_t a[4] = {0}, b[2] = {0}, out[2] = {0};

    zassert_equal(syn_hal_dsp_mat_mult_q15(a, b, out, 2, 2), 0,
                  "Mat mult should succeed");
}
