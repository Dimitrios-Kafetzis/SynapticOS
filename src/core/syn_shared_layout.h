/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file syn_shared_layout.h
 * @brief SynapticOS - Shared SRAM layout for dual-core IPC (private header)
 *
 * Defines the memory map and the in-memory structures shared between
 * CPU0 (AI runtime) and CPU1 (application) on the MCXN947. Both cores
 * compile this header, so every structure here must have an identical
 * layout on both sides; all fields use fixed-width types and explicit
 * padding, and BUILD_ASSERTs pin the offsets.
 *
 * Physical SRAM map (MCXN947 has 416 KB at 0x20000000, RAMA..RAMH,
 * plus 96 KB RAMX at 0x04000000 on the code bus):
 *
 *   0x2000_0000 - 0x2003_FFFF  CPU0 private (256 KB, RAMA-RAME)
 *   0x2004_0000 - 0x2005_7FFF  Shared IPC region (96 KB, RAMF + half RAMG)
 *   0x2005_8000 - 0x2006_7FFF  CPU1 private (64 KB, half RAMG + RAMH)
 *
 * Address views differ per core. CPU0 runs Secure (TrustZone-M) and
 * uses the +0x1000_0000 secure bus aliases: RAM at 0x3xxx_xxxx,
 * flash at 0x1xxx_xxxx (CONFIG_SRAM_BASE_ADDRESS=0x30000000). CPU1
 * is a Cortex-M33 WITHOUT TrustZone (also no FPU/DSP/MPU, per the
 * MCXN947_cm33_core1 CMSIS header) and addresses the same physical
 * memory at the plain addresses (RAM 0x2xxx_xxxx, flash 0x0xxx_xxxx,
 * peripherals 0x4xxx_xxxx). The constants below therefore fold in a
 * per-core alias so each core gets pointers it can dereference.
 *
 * Note: the phase plan sketched a 160 KB CPU1 region ending at
 * 0x2007_FFFF; that exceeds the physical 416 KB main SRAM (96 KB of
 * the chip's 512 KB total is RAMX at 0x0400_0000). CPU1 gets 64 KB of
 * data RAM and executes XIP from flash bank 1.
 *
 * The device tree is the source of truth for the build (linker regions
 * and MPU regions); the constants below exist for runtime validation
 * and for code that must know the map on both cores.
 */
#ifndef SYNAPTIC_SYN_SHARED_LAYOUT_H_
#define SYNAPTIC_SYN_SHARED_LAYOUT_H_

#include <stdint.h>
#include <stddef.h>
#include <zephyr/sys/util.h>
#include <zephyr/toolchain.h>
#include <synaptic/syn_ipc.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Per-core address alias: CPU0 secure view vs CPU1 plain view */
#if defined(CONFIG_SOC_MCXN947_CPU1)
#define SYN_SHM_ALIAS           0x00000000UL
#else
#define SYN_SHM_ALIAS           0x10000000UL
#endif

/* Memory map (MCXN947, this core's view) */
#define SYN_SHM_CPU0_RAM_BASE   (0x20000000UL + SYN_SHM_ALIAS)
#define SYN_SHM_CPU0_RAM_SIZE   (256 * 1024UL)
#define SYN_SHM_SHARED_BASE     (0x20040000UL + SYN_SHM_ALIAS)
#define SYN_SHM_SHARED_SIZE     (96 * 1024UL)
#define SYN_SHM_CPU1_RAM_BASE   (0x20058000UL + SYN_SHM_ALIAS)
#define SYN_SHM_CPU1_RAM_SIZE   (64 * 1024UL)

/* CPU1 image location: XIP from the second half of the 2 MB flash */
#define SYN_SHM_CPU1_FLASH_BASE (0x00100000UL + SYN_SHM_ALIAS)
#define SYN_SHM_CPU1_FLASH_SIZE (1024 * 1024UL)

/* Value CPU0 writes to SYSCON CPBOOT: CPU1 fetches its vector table
 * through its own (non-TrustZone, plain) address view.
 */
#define SYN_BOOT_CPU1_VECTOR    0x00100000UL

/* Layout identification */
#define SYN_SHM_MAGIC           0x53594E33UL /* "SYN3" */
#define SYN_SHM_LAYOUT_VERSION  1UL

/*
 * Ring size seam: CONFIG_SYNAPTIC_IPC_RING_SIZE only exists on
 * dual-core capable builds (gated on SECOND_CORE_MCUX). Single-core
 * builds (QEMU unit tests) compile the same structures with the
 * default entry count so the SPSC logic can be tested anywhere.
 */
#ifdef CONFIG_SYNAPTIC_IPC_RING_SIZE
#define SYN_IPC_RING_ENTRIES    CONFIG_SYNAPTIC_IPC_RING_SIZE
#else
#define SYN_IPC_RING_ENTRIES    16
#endif

/**
 * Shared control block: written by CPU0 during syn_ipc_init(),
 * then used for the boot handshake (deliverable 3.3).
 */
typedef struct {
	uint32_t magic;          /**< SYN_SHM_MAGIC once CPU0 init done   */
	uint32_t layout_version; /**< SYN_SHM_LAYOUT_VERSION              */
	uint32_t ring_entries;   /**< Must match on both cores            */
	uint32_t shared_size;    /**< Region size CPU0 initialized with   */
	volatile uint32_t cpu0_ready; /**< CPU0 runtime + IPC ready       */
	volatile uint32_t cpu1_ready; /**< CPU1 IPC attached              */
	uint32_t reserved[10];   /**< Pad control block to 64 bytes      */
} syn_shm_ctrl_t;

/**
 * Lock-free SPSC ring. Indices are free-running uint32 counters;
 * slot = index % SYN_IPC_RING_ENTRIES, full when head - tail equals
 * the entry count. Only the producer core writes head, only the
 * consumer core writes tail. Each index sits alone in a 64-byte
 * block to avoid false sharing.
 */
typedef struct {
	volatile uint32_t head;  /**< Producer write counter */
	uint32_t _pad0[15];
	volatile uint32_t tail;  /**< Consumer read counter  */
	uint32_t _pad1[15];
	syn_ipc_msg_t slots[SYN_IPC_RING_ENTRIES];
} syn_ipc_ring_t;

/**
 * Full shared region layout. Two rings give each core a dedicated
 * producer role (SPSC in both directions). The payload pool follows
 * and is addressed via syn_ipc_msg_t.payload_offset, which is an
 * offset from the start of the shared region.
 */
typedef struct {
	syn_shm_ctrl_t ctrl;
	syn_ipc_ring_t ring_c0_to_c1; /**< CPU0 produces, CPU1 consumes */
	syn_ipc_ring_t ring_c1_to_c0; /**< CPU1 produces, CPU0 consumes */
	uint8_t payload[];            /**< Rest of the shared region    */
} syn_shm_region_t;

#define SYN_SHM_PAYLOAD_OFFSET  offsetof(syn_shm_region_t, payload)

/*
 * Fixed payload slots for the cross-core inference protocol (3.4).
 * Offsets are relative to the payload pool. Input sized for the
 * largest supported vision input (96x96x3 INT8); output covers
 * classification or detection results with headroom.
 */
#define SYN_SHM_INFER_INPUT_OFFSET   0UL
#define SYN_SHM_INFER_INPUT_SIZE     (27648UL)      /* 96*96*3 */
#define SYN_SHM_INFER_OUTPUT_OFFSET  (SYN_SHM_INFER_INPUT_OFFSET + \
				      SYN_SHM_INFER_INPUT_SIZE)
#define SYN_SHM_INFER_OUTPUT_SIZE    (4096UL)

/* Layout invariants: both cores must agree bit-for-bit. */
BUILD_ASSERT(sizeof(syn_ipc_msg_t) == 20, "syn_ipc_msg_t layout changed");
BUILD_ASSERT(sizeof(syn_shm_ctrl_t) == 64, "control block must be 64 bytes");
BUILD_ASSERT(offsetof(syn_ipc_ring_t, tail) == 64,
	     "head and tail must be in separate 64-byte blocks");
BUILD_ASSERT(offsetof(syn_ipc_ring_t, slots) == 128,
	     "ring slots offset changed");
BUILD_ASSERT(SYN_SHM_PAYLOAD_OFFSET ==
	     sizeof(syn_shm_ctrl_t) + 2 * sizeof(syn_ipc_ring_t),
	     "payload must start right after the rings");

/* The whole layout has to fit the shared region with payload space */
BUILD_ASSERT(SYN_SHM_PAYLOAD_OFFSET +
	     SYN_SHM_INFER_INPUT_SIZE + SYN_SHM_INFER_OUTPUT_SIZE <=
	     SYN_SHM_SHARED_SIZE,
	     "shared region too small for rings plus inference buffers");

/* Map sanity: regions tile 0x20000000..0x20068000 without overlap */
BUILD_ASSERT(SYN_SHM_CPU0_RAM_BASE + SYN_SHM_CPU0_RAM_SIZE ==
	     SYN_SHM_SHARED_BASE, "CPU0 region must abut shared region");
BUILD_ASSERT(SYN_SHM_SHARED_BASE + SYN_SHM_SHARED_SIZE ==
	     SYN_SHM_CPU1_RAM_BASE, "shared region must abut CPU1 region");

#ifdef __cplusplus
}
#endif
#endif /* SYNAPTIC_SYN_SHARED_LAYOUT_H_ */
