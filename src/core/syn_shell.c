/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_shell.c
 * @brief SynapticOS — Zephyr Shell Commands
 *
 * Runtime inspection commands: syn version/mem/model/npu/prof
 */

#include <zephyr/kernel.h>
#include <zephyr/shell/shell.h>
#include <synaptic/syn_api.h>
#include <stdlib.h>

/* syn version */
static int cmd_version(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	shell_print(sh, "SynapticOS v%s", syn_version());
	return 0;
}

/* syn mem stats */
static int cmd_mem_stats(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_mem_stats_t stats;
	int ret = syn_mem_get_stats(&stats);

	if (ret != 0) {
		shell_error(sh, "Failed to get stats: %d", ret);
		return ret;
	}
	shell_print(sh, "Arena: %u/%u bytes (peak %u)",
		    (unsigned)stats.arena_used,
		    (unsigned)stats.arena_total,
		    (unsigned)stats.arena_peak);
	shell_print(sh, "Scratch: %u/%u bytes",
		    (unsigned)stats.scratch_used,
		    (unsigned)stats.scratch_total);
	shell_print(sh, "Allocations: %u, Resets: %u",
		    stats.alloc_count, stats.reset_count);
	return 0;
}

/* syn model list */
static int cmd_model_list(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_model_handle_t handles[CONFIG_SYNAPTIC_MAX_MODELS];
	uint8_t count = 0;

	int ret = syn_model_list(handles, &count, CONFIG_SYNAPTIC_MAX_MODELS);

	if (ret != 0) {
		shell_error(sh, "Failed: %d", ret);
		return ret;
	}

	shell_print(sh, "Registered models: %u", count);
	for (uint8_t i = 0; i < count; i++) {
		syn_model_info_t info;

		syn_model_get_info(handles[i], &info);
		shell_print(sh, "  [%u] %s v%s %s",
			    handles[i], info.name, info.version,
			    syn_model_is_loaded(handles[i]) ? "(loaded)" : "");
	}
	return 0;
}

/* syn npu caps */
static int cmd_npu_caps(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_npu_caps_t caps;
	int ret = syn_hal_npu_get_caps(&caps);

	if (ret != 0) {
		shell_error(sh, "Failed: %d", ret);
		return ret;
	}
	shell_print(sh, "NPU: %s", caps.name);
	shell_print(sh, "  Max OPS/sec: %u", caps.max_ops_per_sec);
	shell_print(sh, "  Scratch: %u bytes", caps.scratch_size);
	shell_print(sh, "  Async: %s", caps.supports_async ? "yes" : "no");
	return 0;
}

/* syn npu state */
static int cmd_npu_state(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	static const char *state_names[] = {"IDLE", "BUSY", "ERROR", "SUSPENDED"};
	syn_npu_state_t state = syn_hal_npu_get_state();

	shell_print(sh, "NPU state: %s", state_names[state]);
	return 0;
}

/* syn prof last */
static int cmd_prof_last(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_prof_result_t result;
	int ret = syn_prof_get_last(&result);

	if (ret != 0) {
		shell_print(sh, "No profiling data available");
		return 0;
	}
	shell_print(sh, "Last inference:");
	shell_print(sh, "  Total:       %u us", result.total_us);
	shell_print(sh, "  Preprocess:  %u us", result.preprocess_us);
	shell_print(sh, "  NPU:         %u us", result.npu_us);
	shell_print(sh, "  Postprocess: %u us", result.postprocess_us);
	shell_print(sh, "  Memory peak: %u bytes", result.mem_peak_bytes);
	return 0;
}

/* syn prof enable */
static int cmd_prof_enable(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_prof_enable();
	shell_print(sh, "Profiling enabled");
	return 0;
}

/* syn prof disable */
static int cmd_prof_disable(const struct shell *sh, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);
	syn_prof_disable();
	shell_print(sh, "Profiling disabled");
	return 0;
}

/* Subcommand trees */
SHELL_STATIC_SUBCMD_SET_CREATE(sub_mem,
	SHELL_CMD(stats, NULL, "Show memory statistics", cmd_mem_stats),
	SHELL_SUBCMD_SET_END
);

SHELL_STATIC_SUBCMD_SET_CREATE(sub_model,
	SHELL_CMD(list, NULL, "List registered models", cmd_model_list),
	SHELL_SUBCMD_SET_END
);

SHELL_STATIC_SUBCMD_SET_CREATE(sub_npu,
	SHELL_CMD(caps, NULL, "Show NPU capabilities", cmd_npu_caps),
	SHELL_CMD(state, NULL, "Show NPU state", cmd_npu_state),
	SHELL_SUBCMD_SET_END
);

SHELL_STATIC_SUBCMD_SET_CREATE(sub_prof,
	SHELL_CMD(last, NULL, "Show last profiling result", cmd_prof_last),
	SHELL_CMD(enable, NULL, "Enable profiling", cmd_prof_enable),
	SHELL_CMD(disable, NULL, "Disable profiling", cmd_prof_disable),
	SHELL_SUBCMD_SET_END
);

SHELL_STATIC_SUBCMD_SET_CREATE(sub_syn,
	SHELL_CMD(version, NULL, "Print SynapticOS version", cmd_version),
	SHELL_CMD(mem, &sub_mem, "Memory management", NULL),
	SHELL_CMD(model, &sub_model, "Model management", NULL),
	SHELL_CMD(npu, &sub_npu, "NPU control", NULL),
	SHELL_CMD(prof, &sub_prof, "Profiling", NULL),
	SHELL_SUBCMD_SET_END
);

SHELL_CMD_REGISTER(syn, &sub_syn, "SynapticOS commands", NULL);
