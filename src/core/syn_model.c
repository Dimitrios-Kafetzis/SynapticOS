/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_model.c
 * @brief SynapticOS — Model Registry
 *
 * Array-backed model registry with duplicate detection, NPU HAL
 * integration for load/unload, and 1-based handle system.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(syn_model, CONFIG_SYNAPTIC_LOG_LEVEL);

#include <synaptic/syn_model.h>
#include <synaptic/syn_hal_npu.h>
#include <string.h>

#ifndef CONFIG_SYNAPTIC_MAX_MODELS
#define CONFIG_SYNAPTIC_MAX_MODELS 8
#endif

struct model_slot {
	bool active;
	bool loaded;
	syn_model_info_t info;
	const uint8_t *model_data;
	size_t model_data_size;
};

static struct model_slot slots[CONFIG_SYNAPTIC_MAX_MODELS];

/* Handle is 1-based index: handle = slot_index + 1 */
static inline int handle_to_index(syn_model_handle_t handle)
{
	if (handle == SYN_MODEL_INVALID || handle > CONFIG_SYNAPTIC_MAX_MODELS) {
		return -1;
	}
	return (int)(handle - 1);
}

int syn_model_register(const syn_model_info_t *info, syn_model_handle_t *handle)
{
	if (info == NULL || handle == NULL) {
		return -EINVAL;
	}

	/* Check for duplicate name */
	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_MODELS; i++) {
		if (slots[i].active &&
		    strncmp(slots[i].info.name, info->name,
			    sizeof(slots[i].info.name)) == 0) {
			LOG_ERR("Model '%s' already registered", info->name);
			*handle = SYN_MODEL_INVALID;
			return -EEXIST;
		}
	}

	/* Find free slot */
	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_MODELS; i++) {
		if (!slots[i].active) {
			slots[i].active = true;
			slots[i].loaded = false;
			slots[i].model_data = NULL;
			slots[i].model_data_size = 0;
			memcpy(&slots[i].info, info, sizeof(syn_model_info_t));
			*handle = (syn_model_handle_t)(i + 1);
			LOG_INF("Registered model '%s' (handle=%u)", info->name, *handle);
			return 0;
		}
	}

	LOG_ERR("Registry full (%d slots)", CONFIG_SYNAPTIC_MAX_MODELS);
	*handle = SYN_MODEL_INVALID;
	return -ENOMEM;
}

int syn_model_unregister(syn_model_handle_t handle)
{
	int idx = handle_to_index(handle);

	if (idx < 0 || !slots[idx].active) {
		return -EINVAL;
	}

	if (slots[idx].loaded) {
		syn_model_unload(handle);
	}

	slots[idx].active = false;
	slots[idx].loaded = false;
	slots[idx].model_data = NULL;
	slots[idx].model_data_size = 0;
	LOG_INF("Unregistered model handle=%u", handle);
	return 0;
}

int syn_model_get_info(syn_model_handle_t handle, syn_model_info_t *info)
{
	int idx = handle_to_index(handle);

	if (idx < 0 || !slots[idx].active || info == NULL) {
		return -EINVAL;
	}

	memcpy(info, &slots[idx].info, sizeof(syn_model_info_t));
	return 0;
}

int syn_model_get_by_name(const char *name, syn_model_handle_t *handle)
{
	if (name == NULL || handle == NULL) {
		return -EINVAL;
	}

	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_MODELS; i++) {
		if (slots[i].active &&
		    strncmp(slots[i].info.name, name, sizeof(slots[i].info.name)) == 0) {
			*handle = (syn_model_handle_t)(i + 1);
			return 0;
		}
	}

	*handle = SYN_MODEL_INVALID;
	return -ENOENT;
}

int syn_model_list(syn_model_handle_t *handles, uint8_t *count, uint8_t max)
{
	if (handles == NULL || count == NULL) {
		return -EINVAL;
	}

	*count = 0;
	for (int i = 0; i < CONFIG_SYNAPTIC_MAX_MODELS && *count < max; i++) {
		if (slots[i].active) {
			handles[*count] = (syn_model_handle_t)(i + 1);
			(*count)++;
		}
	}

	return 0;
}

int syn_model_load(syn_model_handle_t handle)
{
	int idx = handle_to_index(handle);

	if (idx < 0 || !slots[idx].active) {
		return -EINVAL;
	}
	if (slots[idx].loaded) {
		return -EALREADY;
	}

	/* If model data is available, load into NPU HAL */
	if (slots[idx].model_data != NULL && slots[idx].model_data_size > 0) {
		int ret = syn_hal_npu_load_model(slots[idx].model_data,
						 slots[idx].model_data_size);
		if (ret != 0) {
			LOG_ERR("NPU load failed for '%s': %d",
				slots[idx].info.name, ret);
			return ret;
		}
	}

	slots[idx].loaded = true;
	LOG_INF("Loaded model '%s'", slots[idx].info.name);
	return 0;
}

int syn_model_unload(syn_model_handle_t handle)
{
	int idx = handle_to_index(handle);

	if (idx < 0 || !slots[idx].active) {
		return -EINVAL;
	}
	if (!slots[idx].loaded) {
		return -EALREADY;
	}

	slots[idx].loaded = false;
	LOG_INF("Unloaded model '%s'", slots[idx].info.name);
	return 0;
}

bool syn_model_is_loaded(syn_model_handle_t handle)
{
	int idx = handle_to_index(handle);

	if (idx < 0 || !slots[idx].active) {
		return false;
	}

	return slots[idx].loaded;
}

int syn_model_swap(syn_model_handle_t old_handle, syn_model_handle_t new_handle)
{
	int old_idx = handle_to_index(old_handle);
	int new_idx = handle_to_index(new_handle);

	if (old_idx < 0 || new_idx < 0 ||
	    !slots[old_idx].active || !slots[new_idx].active) {
		return -EINVAL;
	}

	slots[old_idx].loaded = false;
	slots[new_idx].loaded = true;
	LOG_INF("Swapped model '%s' -> '%s'",
		slots[old_idx].info.name, slots[new_idx].info.name);
	return 0;
}
