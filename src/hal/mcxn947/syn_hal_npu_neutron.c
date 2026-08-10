/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_hal_npu_neutron.c
 * @brief SynapticOS — eIQ Neutron NPU Driver (MCXN947)
 *
 * Hardware NPU driver for the NXP FRDM-MCXN947 eIQ Neutron NPU.
 *
 * With CONFIG_SYNAPTIC_NEUTRON (optional `neutron` west group
 * providing NXP's eIQ Neutron driver library) models carrying the
 * interim "SYNN" blob header run on the NPU through
 * neutronRunBlocking; anything else (and every build without the
 * option) keeps the software-emulated stub path, including the
 * synthetic layered format.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <synaptic/syn_hal_npu.h>
#include <string.h>

#include "../common/syn_npu_layered.h"

#ifdef CONFIG_SYNAPTIC_NEUTRON
#include <NeutronDriver.h>
#include <fsl_clock.h>
#include <fsl_reset.h>
#endif

LOG_MODULE_REGISTER(syn_hal_npu_neutron, CONFIG_SYNAPTIC_LOG_LEVEL);

/* Stub-inference bound: the HAL only keeps an XIP pointer, so accept
 * anything a flash model slot can hold (440 KB minus the .synm
 * header). Board-found: the old arbitrary 256 KB cap rejected a
 * valid slot-max OTA model AFTER it was stored and activated. The
 * real bound comes with the Neutron SDK invoke path.
 */
#define NEUTRON_MAX_MODEL_SIZE    (440 * 1024 - 64)
#define NEUTRON_MAX_INPUT_SIZE    (96 * 96 * 3)
#define NEUTRON_MAX_OUTPUT_SIZE   256

#ifdef CONFIG_SYNAPTIC_NEUTRON
/* Interim Neutron blob format ("SYNN", little-endian), single
 * input / single output. neutron-converter emits a microcode /
 * weights / kernels triple; the packer concatenates them behind
 * this header with every section 16-byte aligned relative to the
 * blob start (the store hands out 16-byte-aligned XIP pointers, so
 * relative alignment is enough). The proper .synm container mapping
 * replaces this in 6.1c.
 */
#define SYN_NEUTRON_MAGIC 0x4E4E5953UL /* "SYNN" */

struct syn_neutron_hdr {
	uint32_t magic;
	uint32_t microcode_off;
	uint32_t microcode_size;
	uint32_t weights_off;
	uint32_t weights_size;
	uint32_t kernels_off;
	uint32_t kernels_size;
	uint32_t input_size;
	uint32_t output_size;
};

static struct {
	bool               hw_ok;       /* neutronInit succeeded */
	bool               prepared;    /* hdl refers to a live model */
	NeutronModelHandle hdl;
	/* BOARD FINDING (Phase 6): the driver stores &mcfg in the model
	 * handle and dereferences it again on every neutronRunBlocking
	 * (handle[0]->microcode feeds the firmware interpreter). A
	 * stack-local config therefore fails at run time with
	 * Firmware/microcode code 134 once the prepare frame is gone -
	 * the config must live as long as the prepared model.
	 */
	NeutronModelConfig mcfg;
	uint32_t           input_size;  /* from the SYNN header */
	uint32_t           output_size;
	uint8_t            ctx[CONFIG_SYNAPTIC_NEUTRON_CTX_MAX] __aligned(16);
	uint8_t            scratch[CONFIG_SYNAPTIC_NEUTRON_SCRATCH_SIZE] __aligned(16);
} neutron;

/* BOARD FINDING (Phase 6): like the eDMA (see syn_hal_dma_edma.c),
 * the Neutron NPU masters its own bus transactions and cannot use
 * CPU0's TrustZone secure aliases (bit 28): microcode fetched via a
 * 0x10xxxxxx flash pointer reads as garbage and the firmware rejects
 * it as "bad magic" (Firmware/microcode code 134). Every address
 * handed to the driver must be the plain alias.
 */
static const void *npu_addr(const void *p)
{
	return (const void *)((uintptr_t)p & ~BIT(28));
}

static void neutron_log_error(const char *what, NeutronError e)
{
	LOG_ERR("%s: %s/%s code %ld", what,
		getNeutronErrorComponent(e), getNeutronErrorCategory(e),
		(long)GET_ERROR_CODE(e));
}

static int neutron_unprepare(void)
{
	if (!neutron.prepared) {
		return 0;
	}

	NeutronError e = neutronModelUnprepare(neutron.hdl);

	neutron.prepared = false;
	neutron.hdl = NEUTRON_INVALID_HANDLE;
	if (e != ENONE) {
		neutron_log_error("neutronModelUnprepare", e);
		return -EIO;
	}
	return 0;
}

/* Returns 0 when the blob is a valid SYNN model and it was prepared
 * on the NPU, -ENOTSUP when the blob is not a SYNN model (caller
 * keeps the stub path), a negative errno otherwise.
 */
static int neutron_prepare(const uint8_t *data, size_t size)
{
	const struct syn_neutron_hdr *hdr = (const void *)data;

	if (size < sizeof(*hdr) || hdr->magic != SYN_NEUTRON_MAGIC) {
		return -ENOTSUP;
	}
	if (!neutron.hw_ok) {
		LOG_ERR("SYNN model but the NPU is unavailable");
		return -ENODEV;
	}
	if ((uint64_t)hdr->microcode_off + hdr->microcode_size > size ||
	    (uint64_t)hdr->weights_off + hdr->weights_size > size ||
	    (uint64_t)hdr->kernels_off + hdr->kernels_size > size) {
		LOG_ERR("SYNN section out of bounds (blob %zu B)", size);
		return -EINVAL;
	}
	if ((hdr->microcode_off | hdr->weights_off | hdr->kernels_off) & 0xF) {
		LOG_ERR("SYNN section misaligned (16-byte required)");
		return -EINVAL;
	}
	if (hdr->input_size == 0 ||
	    hdr->input_size > NEUTRON_MAX_INPUT_SIZE ||
	    hdr->output_size == 0 ||
	    hdr->output_size > NEUTRON_MAX_OUTPUT_SIZE) {
		LOG_ERR("SYNN i/o sizes unsupported (%u/%u)",
			hdr->input_size, hdr->output_size);
		return -EINVAL;
	}

	size_t ctx_need = neutronGetModelContextSize();

	if (ctx_need > sizeof(neutron.ctx)) {
		LOG_ERR("Neutron context needs %zu B, buffer is %zu B "
			"(raise SYNAPTIC_NEUTRON_CTX_MAX)",
			ctx_need, sizeof(neutron.ctx));
		return -ENOMEM;
	}

	int rc = neutron_unprepare();

	if (rc != 0) {
		return rc;
	}

	neutron.mcfg = (NeutronModelConfig){
		.microcode      = npu_addr(data + hdr->microcode_off),
		.weights        = npu_addr(data + hdr->weights_off),
		.kernels        = npu_addr(data + hdr->kernels_off),
		.timeoutSeconds = 10,
		.subgraphName   = NULL,
	};

	neutron.hdl = (NeutronModelHandle)neutron.ctx;

	NeutronError e = neutronModelPrepare(&neutron.mcfg, &neutron.hdl);

	if (e != ENONE) {
		neutron.hdl = NEUTRON_INVALID_HANDLE;
		neutron_log_error("neutronModelPrepare", e);
		return -EIO;
	}

	neutron.prepared = true;
	neutron.input_size = hdr->input_size;
	neutron.output_size = hdr->output_size;
	LOG_INF("Neutron model prepared: ucode %u B, weights %u B, "
		"kernels %u B, io %u/%u B, ctx %zu B",
		hdr->microcode_size, hdr->weights_size, hdr->kernels_size,
		hdr->input_size, hdr->output_size, ctx_need);
	return 0;
}
#endif /* CONFIG_SYNAPTIC_NEUTRON */

static struct {
	syn_npu_state_t state;
	bool            model_loaded;
	size_t          model_size;
	const uint8_t  *model_data;
	uint8_t         input_buf[NEUTRON_MAX_INPUT_SIZE];
	size_t          input_size;
	uint8_t         output_buf[NEUTRON_MAX_OUTPUT_SIZE];
	size_t          output_size;
	bool            initialized;
} npu;

int syn_hal_npu_init(void)
{
	if (npu.initialized) {
		return -EALREADY;
	}

	memset(&npu, 0, sizeof(npu));

#ifdef CONFIG_SYNAPTIC_NEUTRON
	memset(&neutron, 0, sizeof(neutron));
	CLOCK_EnableClock(kCLOCK_Neutron);
	RESET_ClearPeripheralReset(kNEUTRON_RST_SHIFT_RSTn);

	NeutronError e = neutronInit();

	if (e == ENONE) {
		neutron.hw_ok = true;
		LOG_INF("Neutron NPU online (driver initialized)");
	} else {
		/* Same policy as the PowerQuad HAL: a failed bring-up
		 * degrades to the software path instead of taking the
		 * whole runtime down. SYNN models are rejected at load.
		 */
		CLOCK_DisableClock(kCLOCK_Neutron);
		neutron_log_error("neutronInit", e);
		LOG_WRN("Neutron unavailable, stub inference only");
	}
#endif

	npu.state = SYN_NPU_STATE_IDLE;
	npu.initialized = true;

	LOG_INF("Neutron NPU HAL initialized");
	return 0;
}

void syn_hal_npu_deinit(void)
{
#ifdef CONFIG_SYNAPTIC_NEUTRON
	if (neutron.hw_ok) {
		(void)neutron_unprepare();

		NeutronError e = neutronDeinit();

		if (e != ENONE) {
			neutron_log_error("neutronDeinit", e);
		}
		CLOCK_DisableClock(kCLOCK_Neutron);
		neutron.hw_ok = false;
	}
#endif

	npu.initialized = false;
	npu.state = SYN_NPU_STATE_IDLE;
	npu.model_loaded = false;
	LOG_INF("Neutron NPU deinitialized");
}

int syn_hal_npu_get_caps(syn_npu_caps_t *caps)
{
	if (caps == NULL) {
		return -EINVAL;
	}

	caps->name = "neutron";
	/* MCXN947 Neutron NPU: ~100 GOPS at INT8 */
	caps->max_ops_per_sec = 100000000;
	caps->scratch_size = CONFIG_SYNAPTIC_SCRATCH_POOL_SIZE;
	caps->supported_dtypes = 0x03; /* INT8 and UINT8 for Neutron */
	caps->supports_async = true;

	return 0;
}

syn_npu_state_t syn_hal_npu_get_state(void)
{
	return npu.state;
}

int syn_hal_npu_load_model(const uint8_t *model_data, size_t model_size)
{
	if (model_data == NULL || model_size == 0) {
		return -EINVAL;
	}
	if (!npu.initialized) {
		return -EPERM;
	}
	if (npu.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}
	if (model_size > NEUTRON_MAX_MODEL_SIZE) {
		return -ENOMEM;
	}

#ifdef CONFIG_SYNAPTIC_NEUTRON
	int rc = neutron_prepare(model_data, model_size);

	if (rc == 0) {
		npu.model_data = model_data;
		npu.model_size = model_size;
		npu.model_loaded = true;
		return 0;
	}
	if (rc != -ENOTSUP) {
		return rc;
	}
	/* Not a SYNN blob: a previously prepared Neutron model no
	 * longer matches the loaded one; drop it and use the stub path.
	 */
	(void)neutron_unprepare();
#endif

	npu.model_data = model_data;
	npu.model_size = model_size;
	npu.model_loaded = true;
	syn_npu_layered_on_load(model_data, model_size);

	LOG_INF("Model loaded: %zu bytes (stub inference)", model_size);
	return 0;
}

int syn_hal_npu_set_input(uint8_t index, const void *data, size_t size)
{
	if (data == NULL || size == 0 || index != 0) {
		return -EINVAL;
	}
	if (!npu.initialized || !npu.model_loaded) {
		return -EPERM;
	}
	if (npu.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}
	if (size > NEUTRON_MAX_INPUT_SIZE) {
		return -ENOMEM;
	}

	/* TODO: Set input via Neutron SDK:
	 *   neutron_set_input(index, data, size);
	 */
	memcpy(npu.input_buf, data, size);
	npu.input_size = size;

	return 0;
}

int syn_hal_npu_invoke(void)
{
	if (!npu.initialized || !npu.model_loaded) {
		return -EPERM;
	}
	if (npu.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}

	npu.state = SYN_NPU_STATE_BUSY;

#ifdef CONFIG_SYNAPTIC_NEUTRON
	if (neutron.prepared) {
		if (npu.input_size != neutron.input_size) {
			npu.state = SYN_NPU_STATE_IDLE;
			LOG_ERR("Input is %zu B, model expects %u B",
				npu.input_size, neutron.input_size);
			return -EINVAL;
		}

		/* BOARD FINDING (Phase 6): the firmware ABI indexes past
		 * the declared data outputs - the converted graph's output
		 * list is {data, scratch, profile, debug} and the driver
		 * reads outputs[numOutputs] and outputs[numOutputs+1]
		 * unconditionally. A 1-entry array therefore feeds stack
		 * garbage to the firmware as write pointers, which sprays
		 * inference data over the thread stacks (silent boot-time
		 * crash, no fault dump). Slot 1 must be the scratch buffer;
		 * the shape-0 profile/debug slots take NULL.
		 */
		const void *inputs[1] = { npu_addr(npu.input_buf) };
		void *outputs[4] = {
			(void *)npu_addr(npu.output_buf),
			(void *)npu_addr(neutron.scratch),
			NULL,
			NULL,
		};
		NeutronDataConfig dcfg = {
			.inputs         = inputs,
			.outputs        = outputs,
			.scratch        = (void *)npu_addr(neutron.scratch),
			.scratchWeights = NULL,
		};

		LOG_DBG("Neutron run: enter (ucode %p)", neutron.mcfg.microcode);

		NeutronError e = neutronRunBlocking(neutron.hdl, &dcfg);

		LOG_DBG("Neutron run: exit (%d)", (int)e);

		if (e != ENONE) {
			npu.state = SYN_NPU_STATE_ERROR;
			neutron_log_error("neutronRunBlocking", e);
			return -EIO;
		}

		npu.output_size = neutron.output_size;
		npu.state = SYN_NPU_STATE_IDLE;
		LOG_DBG("Neutron inference complete (%u B out)",
			neutron.output_size);
		return 0;
	}
#endif

	/* Stub inference: deterministic 10-class classification */
	npu.output_size = 10;
	memset(npu.output_buf, 0, npu.output_size);

	uint32_t sum = 0;

	for (size_t i = 0; i < npu.input_size; i++) {
		sum += npu.input_buf[i];
	}

	uint8_t winner = sum % npu.output_size;

	npu.output_buf[winner] = 127;

	/* Simulate NPU inference latency */
	k_busy_wait(1000);

	npu.state = SYN_NPU_STATE_IDLE;

	LOG_DBG("Neutron inference complete (stub): class %u", winner);
	return 0;
}

int syn_hal_npu_invoke_async(syn_npu_done_cb_t cb, void *user_data)
{
	/* TODO: Implement async invoke using Neutron NPU interrupt */
	ARG_UNUSED(cb);
	ARG_UNUSED(user_data);

	return -ENOTSUP;
}

int syn_hal_npu_get_output(uint8_t index, void *data, size_t *size)
{
	if (data == NULL || size == NULL || index != 0) {
		return -EINVAL;
	}
	if (!npu.initialized) {
		return -EPERM;
	}
	if (npu.state == SYN_NPU_STATE_BUSY) {
		return -EBUSY;
	}

	/* TODO: Get output via Neutron SDK:
	 *   neutron_get_output(index, data, size);
	 */
	memcpy(data, npu.output_buf, npu.output_size);
	*size = npu.output_size;

	return 0;
}

int syn_hal_npu_suspend(void)
{
	if (!npu.initialized) {
		return -EPERM;
	}

#ifdef CONFIG_SYNAPTIC_NEUTRON
	if (neutron.hw_ok) {
		NeutronError e = neutronSuspend();

		if (e != ENONE) {
			neutron_log_error("neutronSuspend", e);
			return -EIO;
		}
	}
#endif

	npu.state = SYN_NPU_STATE_SUSPENDED;
	LOG_DBG("Neutron NPU suspended");
	return 0;
}

int syn_hal_npu_resume(void)
{
	if (!npu.initialized) {
		return -EPERM;
	}
	if (npu.state != SYN_NPU_STATE_SUSPENDED) {
		return -EINVAL;
	}

#ifdef CONFIG_SYNAPTIC_NEUTRON
	if (neutron.hw_ok) {
		NeutronError e = neutronResume();

		if (e != ENONE) {
			neutron_log_error("neutronResume", e);
			return -EIO;
		}
	}
#endif

	npu.state = SYN_NPU_STATE_IDLE;
	LOG_DBG("Neutron NPU resumed");
	return 0;
}
