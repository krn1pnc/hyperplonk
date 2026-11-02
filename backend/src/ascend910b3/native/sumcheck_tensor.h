#pragma once
// Private host-safe Cube-message SumCheck interface for the degree-2 natural
// route. Canonical 32-byte Fr records; no allocation, synchronization, or host
// arithmetic in any launcher.
#include <cstdint>
#include <acl/acl.h>
#include "field.h"

namespace hyperplonk::ascend910b3::detail {
inline constexpr uint32_t sumcheck_tensor_tile = 512;
inline constexpr uint32_t sumcheck_tensor_cell = 32;
inline constexpr uint32_t sumcheck_tensor_redc_lanes = 16;
inline constexpr uint32_t sumcheck_tensor_batch_tiles = 16;
inline constexpr uint32_t sumcheck_tensor_max_groups = 20;
inline constexpr uint32_t sumcheck_tensor_degree = 2;
inline constexpr uint32_t sumcheck_tensor_partial_slots = 3;
inline constexpr uint64_t sumcheck_tensor_raw_c_offset = 65536;
inline constexpr uint64_t sumcheck_tensor_sums_offset = 327680;
inline constexpr uint64_t sumcheck_tensor_corner_sums_offset = 335872;
// One message-input slot is the original 64 KiB four-corner operand block:
// four 16 KiB int8 A/B operands, one per corner (K512 each). Slot 0 aliases
// this block at the start of every group record; remaining slots are appended
// after the record, so no input slot can alias raw C, sums or corner sums.
inline constexpr uint32_t sumcheck_tensor_input_slots = 2;
inline constexpr uint64_t sumcheck_tensor_input_slot_bytes = 65536;
inline constexpr uint64_t sumcheck_tensor_group_bytes = 335936;
inline constexpr uint64_t sumcheck_tensor_input_alt_offset = sumcheck_tensor_group_bytes;
inline constexpr uint64_t sumcheck_tensor_group_stride =
    sumcheck_tensor_input_alt_offset +
    uint64_t(sumcheck_tensor_input_slots - 1) * sumcheck_tensor_input_slot_bytes;
static_assert(sumcheck_tensor_input_slot_bytes == 4ull * 16384,
              "an input slot must hold four corner operands");
static_assert(sumcheck_tensor_group_bytes ==
                  sumcheck_tensor_raw_c_offset + 4ull * 16 * 4096 +
                      4ull * 16 * 128 + 64,
              "the first input slot must fill the original group record");
static_assert(sumcheck_tensor_group_stride == 401472 &&
                  sumcheck_tensor_group_stride % 32 == 0,
              "per-group workspace stride changed");
// Actual device allocation: shared compact engine + passed wide-conversion
// buffers + packing/correction lifetime-shared scratch, all disjoint otherwise.
inline constexpr uint64_t sumcheck_tensor_ub_user_bytes = 177696;
inline constexpr uint64_t sumcheck_tensor_sdk_reserve_bytes = 8192;
inline constexpr uint64_t sumcheck_tensor_l1_bytes = 65536;
inline constexpr uint64_t sumcheck_tensor_l0a_bytes = 16384;
inline constexpr uint64_t sumcheck_tensor_l0b_bytes = 16384;
inline constexpr uint64_t sumcheck_tensor_l0c_bytes = 4096;
static_assert(sumcheck_tensor_ub_user_bytes <= 188160);
static_assert(sumcheck_tensor_ub_user_bytes + sumcheck_tensor_sdk_reserve_bytes <= 196352);
static_assert(sumcheck_tensor_l1_bytes <= 524032);
static_assert(sumcheck_tensor_l0a_bytes <= 65536 && sumcheck_tensor_l0b_bytes <= 65536);
static_assert(sumcheck_tensor_l0c_bytes <= 131072);

uint64_t sumcheck_tensor_workspace_bytes(uint32_t max_groups);
bool launch_sumcheck_tensor_message_d2(aclrtStream stream,
    const uint64_t* absolute_offsets, uint64_t rest_count, Fr* partial,
    uint32_t product_index, uint32_t blocks, void* workspace);
} // namespace hyperplonk::ascend910b3::detail
