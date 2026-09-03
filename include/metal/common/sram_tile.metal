#pragma once

#include <metal_stdlib>
using namespace metal;

namespace metal_llm {
namespace common {

constant constexpr uint SRAM_PAD_STRIDE_36 = 36;
constant constexpr uint SRAM_K_STEP_32     = 32;
constant constexpr uint SRAM_M_TILE_64     = 64;
constant constexpr uint SRAM_N_TILE_64     = 64;

typedef half PaddedTileA[SRAM_M_TILE_64][SRAM_PAD_STRIDE_36];
typedef half WeightTileB[SRAM_K_STEP_32][SRAM_N_TILE_64];

} // namespace common
} // namespace metal_llm
