# CuTe DSL FMHA Architecture: Comparison & Migration Plan

## Overview

This document compares three implementations of Blackwell (SM100) Flash Multi-Head Attention using CuTe DSL / CUTLASS, and outlines a migration plan to modularize FlashInfer's CuTe DSL kernels for maintainability, extensibility, and performance.

### Implementations Compared

| Implementation | Location | Language | Role |
|----------------|----------|----------|------|
| **C++ CUTLASS** | `cutlass/examples/77_blackwell_fmha/` | C++ templates | Upstream reference; most complete |
| **DKG Python** | `cutlass_ir/compiler/python/examples/blackwell/` | Python CuTe DSL | Faithful port with additional optimizations |
| **FlashInfer PR #1549** | `flashinfer/cute_dsl/` | Python CuTe DSL | Derivative with user-facing customization hooks |

---

## 1. PR #1549 Changes Summary

PR #1549 ("[CuTe DSL] Add Blackwell MHA prefill and MLA decode kernel") adds two new high-performance attention kernels for NVIDIA Blackwell (SM100) GPUs written in CuTe DSL.

### New Files

- **`flashinfer/cute_dsl/prefill.py`** (~2933 lines) — Fused MHA prefill kernel
- **`flashinfer/cute_dsl/mla.py`** (~3641 lines) — Multi-Latent Attention decode kernel
- **`flashinfer/cute_dsl/patch/pipeline.py`** (~419 lines) — Producer/consumer pipeline wrappers

### MHA Prefill Kernel

- Warp specialization with 16 warps: 8 softmax (4+4 for double-buffered Q tiles), 4 correction, 1 MMA, 1 load, 1 epilogue, 1 empty
- Multi-stage pipelining: TMA load, MMA, softmax, correction, epilogue overlap via async barriers
- Uses Tensor Memory (TMEM) for intermediate QK scores and PV accumulators
- Double-buffered Q tiles (2 per CTA) for maximum throughput
- Customizable: logits transforms, output transforms, attention sink support, sliding window, causal masking
- Reaches ~1247 TFLOPs on B200 at seq_len=65536

### MLA Decode Kernel

- Split-KV attention for long sequences with separate reduction kernel
- Paged KV cache with page tables
- Separate latent (512d) and RoPE (64d) dimension handling
- Configurable multi-CTA clusters
- Variable sequence lengths and variable split-KV
- Reaches ~1436 TFLOPs and ~6116 GB/s on B200

### Attention Sinks

The prefill kernel supports attention sinks via user-injectable `@cute.jit` callbacks:

- **`M_D_update`**: Modifies online softmax's running max and sum to account for a per-head sink value on the first KV tile
- **`output_transform`**: Rescales the output by the corrected normalizer

The sink value acts as a virtual extra token in the softmax denominator:
$$p_i = \frac{\exp(s_i)}{\sum_j \exp(s_j) + \text{sink}_h}$$

This is implemented by injecting `log(sink)` into the online softmax at `kv_tile_idx == 0`, with zero overhead on subsequent tiles (compiles away via `cutlass.const_expr`).

### Other Changes

- **Routed MoE API**: `trtllm_fp4_block_scale_routed_moe()` accepting pre-computed top-k routing decisions
- **TRT-LLM headers**: Extended batched GEMM and fused MoE for routing maps
- **Benchmarks**: New benchmarks for CuTe DSL prefill and MLA decode
- **Tests**: Comprehensive test coverage including variable-length, GQA, and attention sink variants

---

## 2. Prefill Kernel vs. Decode Kernel (within PR #1549)

| Aspect | Prefill (`prefill.py`) | Decode MLA (`mla.py`) |
|--------|------------------------|-----------------------|
| **Workload** | Standard MHA for long sequences | Multi-Latent Attention for decoding |
| **Q shape** | Many Q tokens (seq_len 512–65K) | One Q token per sequence |
| **Bottleneck** | Compute-bound | Memory-bound |
| **Warps** | 16 (fine-grained specialization) | 6–8 (merged compute) |
| **Pipelines** | 10 (deeply pipelined) | 5 (simpler) |
| **Q double-buffering** | Yes (2 Q tiles/CTA) | No |
| **Split-KV** | No | Yes (+ reduction kernel) |
| **KV access** | Contiguous tensors | Paged with page tables |
| **Q/K structure** | Monolithic | Latent + RoPE decomposed |
| **Clusters** | Single CTA | Multi-CTA supported |
| **Customization** | logits/output transforms, sinks | Fixed-function |

### Warp Layout Comparison

**Prefill (16 warps):**
| Warps | Role | Registers |
|-------|------|-----------|
| 0–3 | Softmax0 (Q tile 0) | 192 |
| 4–7 | Softmax1 (Q tile 1) | 192 |
| 8–11 | Correction | 96 |
| 12 | MMA | 32 |
| 13 | Load | 32 |
| 14 | Epilogue | 32 |
| 15 | Empty | 24 |

**Decode MLA (6–8 warps):**
| Warps | Role | Registers |
|-------|------|-----------|
| 0–3 | Compute (softmax + correction merged) | 192 |
| 4 | MMA | 96 |
| 5 | TMA Load | 96 |
| 6 | Page Table Load (optional) | 96 |
| 7 | Empty (optional) | 96 |

---

## 3. FlashInfer vs. DKG Python vs. C++ CUTLASS

### Prefill Kernel

| Aspect | C++ CUTLASS | DKG Python | FlashInfer |
|--------|-------------|------------|------------|
| **Warp layout** | 16 warps, same roles | 16 warps, same roles | 16 warps, same roles |
| **TMEM layout** | S0@0, S1@128, O0@256, O1@384, P in S | Same | Same |
| **Pipelines** | 8 + `OrderedSequenceBarrier` | 12 (explicit P0/P1 + inplace) | 10 (using `pipeline_patch.py`) |
| **P storage** | TMEM (PV reads P from TMEM) | TMEM | TMEM |
| **Masking** | Template policy (`CausalMask<bool>`) with 3-phase loop | `FusedMask` class, similar 3-phase | `MaskType` enum, similar splitting |
| **Softmax pipelining** | Manual SW pipeline: `kFmaPipeCount=8`, `kConvertPipeCount=16` | Not as deeply pipelined | Not as deeply pipelined |
| **exp2 emulation** | `enable_exp2_emulation` flag | Not mentioned | Not present |
| **Skip-correction** | Not shown in prefill | `enable_skip_correction` | Not present |
| **FP8** | Full (E4M3/E5M2 with scale_q/k/v/o) | Full (E4M3) | BF16/FP16 only |
| **LSE output** | Optional `lse_calculation=True` | Optional | Not exposed |
| **Customization** | Static template mask policies | Fixed-function | **Logits/output transforms, attention sinks** |
| **Variable-length** | `VariableLength` with cumulative_length | `cum_seqlen_q/k` | `cum_seqlen_q/k` |
| **Persistent mode** | Yes | Yes | Yes |
| **Backward pass** | Yes | No | No |
| **Code organization** | 5+ files (kernel, mainloop, loader, epilogue, common) | 1 file (~3600 lines) | 1 file (~2930 lines) |

### Decode / MLA Kernel

| Aspect | C++ CUTLASS MLA | DKG Python MLA | FlashInfer MLA |
|--------|-----------------|----------------|----------------|
| **Warps** | 8 (4 compute, 1 MMA, 1 TMA, 1 PT, 1 empty) | 12 (4 compute, **4 correction**, 1 MMA, 1 TMA, 1 PT, 1 empty) | 6–8 (4 compute, 1 MMA, 1–2 load) |
| **Correction** | Merged into compute warps | **Separate 4-warp group** (208 regs) | Merged into compute warps |
| **2-CTA cluster** | Always (`kIs2Sm=true`) | Always (`cluster=(2,1,1)`) | Configurable |
| **KV pipeline stages** | 12–15 (unified K+V) | 15 (unified K+V) | Separate load_q/load_kv |
| **PT load** | Dedicated warp + `PipelineCpAsync` (4 stages) | Dedicated warp + 4-stage pipeline | Optional, inline in load warp |
| **Split-KV reduction** | Separate + **fused atomic** (`atomicMax` + `TMA_REDUCE_ADD`) | Separate only | Separate only |
| **Skip-correction** | Not shown | Yes (`vote_all_sync`) | Not present |
| **FP8 variant** | Separate template instantiation | Separate file (`mla_decode_fp8.py`) | Not present |
| **Paged KV** | TMA + CpAsync with `ComposedLayout<Gather>` | TMA with page table | TMA with page table |

### Gen (Decode) Kernel — C++ Only

The C++ CUTLASS implementation includes a **gen kernel** (`sm100_fmha_gen_*`) for the generation/decode phase with features not present in either Python version:

- **KV cache append**: Fuses loading new K/V, appending to cache, and attention into one kernel
- **`cp.async` load path**: Software-managed async copies (vs. TMA) enabling complex access patterns for per-batch variable-length KV, cache index remapping, and cache+new stitching
- **GQA head packing**: Maps multiple Q heads into the MMA N-dimension (groups of 8/16/32)
- **Direct GMEM epilogue**: Correction warp writes O directly to global memory (no TMA store needed for single-token output)

### Mixed-Input FMHA — DKG Python Only

The DKG implementation includes mixed-precision variants not present elsewhere:

- **`mixed_input_fmha_decode.py`**: Q in BF16, K/V in Int8/Int4/FP8 with per-block scale factors
- **`mixed_input_fmha_prefill_d256.py`** / **`d512.py`**: Prefill with 8 dedicated transform warps for KV dequantization
- Dequant path: load quantized KV → multiply by block scales → write to SMEM/TMEM as BF16

---

## 4. Structural Comparison: C++ Templates vs. Python JIT

| | C++ CUTLASS | Python CuTe DSL |
|---|---|---|
| **Dispatch** | AOT — every (dtype, headdim, mask, schedule) combo is a separate template instantiation | JIT — compile only the exact config needed at runtime |
| **Binary size** | Large (combinatorial explosion) | Minimal (one .so per runtime config) |
| **Compile time** | Minutes to hours for full build | Seconds per kernel variant |
| **Performance ceiling** | Maximum — all decisions compile-time, zero dispatch overhead | Equivalent kernel quality (same PTX/SASS) |
| **Customization** | Static template policies — hard to add new variants | Runtime-composable — can inject arbitrary Python callables |
| **Readability** | Heavy template syntax; verbose pipeline management | Algorithm maps more directly to the math |
| **Error messages** | Opaque template errors | Python tracebacks |
| **CUTLASS infra reuse** | Full (`CollectiveBuilder`, standard pipelines, `TiledMma`) | Partial (CuTe layout algebra, MMA atoms, but not `CollectiveBuilder`) |

---

## 5. Variable Head Count: Mapping Heads to MMA Dimensions

### The core issue: prefill vs. decode head mapping

The way attention heads are mapped to MMA dimensions differs fundamentally between prefill and decode kernels, and this has major implications for supporting different head counts (128, 64, 32, 16, 8, ...) and attention variants (MQA, GQA, MLA).

#### Prefill kernels: heads in the grid dimension

In **all three implementations**, prefill kernels map heads to the outermost "L" (loop/batch) dimension, not to the MMA tile. Looking at FlashInfer's prefill layout:

```python
# (s, d, ((h_r, h_k), b))
q_layout = cute.make_layout(
    (s_q_all, d, ((h_r, h_k), b_q)),
    stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_q)),
)
```

- **M-dimension**: `s_q` (sequence length) → MMA M-tile
- **K-dimension**: `d` (head dimension) → MMA K-tile
- **L-dimension**: `((h_r, h_k), b)` (heads × batch) → **grid dimension**

Each CTA processes one `(seq_q_tile, head_group)` pair. The number of heads simply determines how many CTAs launch along the grid Y and Z dimensions. **Any number of heads works naturally** for prefill — it just changes the grid size, not the MMA tile utilization. The same applies to training/backward kernels.

#### MLA decode kernels: heads in the MMA M-dimension

In MLA decode, Q has `seq_len = 1`, so the sequence dimension collapses. Instead, **all heads are packed into the MMA M-dimension**:

```
MMA shape: (M=num_heads, N=seq_len_k_tile, K=rope_dim)
S = Q_all_heads * K_tile^T  →  shape (num_heads, kv_tile_size)
```

One CTA processes **all heads simultaneously** in a single MMA operation. The MMA tile M-dimension is fixed at 128 (the tcgen05 instruction width). When `num_heads = 128`, it fits perfectly. When `num_heads < 128`, the extra rows are "phantom" heads — they compute but their results are discarded.

This design makes sense because:
1. With only 1 query token, the sequence dimension can't fill the MMA tile
2. DeepSeek MLA's compressed KV cache is shared across all heads (no per-head K/V), so all heads naturally process the same KV data
3. Packing heads into the M-dimension maximizes MMA utilization for the common 128-head case

#### Standard (non-MLA) decode: GQA groups in the MMA N-dimension

The DKG `fmha_decode.py` handles standard GQA decode differently from MLA — it packs **GQA groups** into the MMA N-dimension:

```
grouped_head_tile = min(num_heads_q / num_heads_kv, 32)  # e.g., 8, 16, or 32
MMA shape: (M=kv_tile, N=grouped_head_tile, K=head_dim)
```

The head ratio (GQA group size) determines how many Q-heads share the same K/V tile. Typical GQA ratios (4, 8, 16, 32) fit naturally into the N-dimension.

### Cross-implementation comparison: variable head support

| Kernel | Head mapping | Variable heads? | Why |
|--------|-------------|----------------|-----|
| **Prefill (all impls)** | Heads → grid dimension | Naturally supported | Grid size just changes |
| **Training/backward** | Same as prefill | Naturally supported | Grid size just changes |
| **Standard decode (DKG)** | GQA groups → MMA N-dim | Works for typical GQA ratios | GQA ratio is usually small (4–32) |
| **MLA decode (C++ CUTLASS)** | All heads → MMA M-dim | **Only 128 heads** | `static_assert(TileShapeH{} == 128)` |
| **MLA decode (DKG Python)** | All heads → MMA M-dim | **Only 128 heads** | `mma_qk_tiler_mn[0] != 128` check |
| **MLA decode (FlashInfer)** | All heads → MMA M-dim | **128, 64, 32, 16, 8** | Runtime `num_heads` parameter with boundary masking |

### How FlashInfer handles variable heads in MLA decode

FlashInfer's MLA kernel makes `num_heads` a **runtime parameter** rather than a compile-time constant:

1. **Conditional 2-CTA mode** — only use cooperative 2-CTA instructions when `num_heads == 128`:
   ```python
   self._use_2cta_instrs = num_heads == 128
   ```
   For fewer heads, fall back to single-CTA mode.

2. **Over-provisioned MMA tile** — always use 128-wide M-tiles, but track actual head count:
   ```python
   cta_qk_tiler = (
       self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],  # e.g., 128 or 64 per CTA
       self.mma_qk_tiler[1],
       self.mma_qk_tiler[2],
   )
   ```

3. **Boundary masking in epilogue** — only write results for real heads:
   ```python
   if cute.elem_less(tTR_cO[0][0], self.num_heads):
       cute.autovec_copy(tR2G_rO_src, tR2G_rO_dst)
   ```

4. **LSE output masking**:
   ```python
   if cute.elem_less(cLSE[common_params.tidx][0], self.num_heads):
       gLSE[common_params.tidx] = lse
   ```

**Performance tradeoff**: For `num_heads < 128`, some compute is wasted on phantom head rows. But MLA decode is **memory-bandwidth-bound** (not compute-bound), so the wasted compute is hidden by KV cache load latency. The PR's performance table confirms this — going from 128 to 8 heads, TFLOPs drops proportionally (684→43) but memory bandwidth stays high (3541→2895 GB/s).

### Can `TileBounds` handle MQA, GQA, and MLA decode uniformly?

The `TileBounds` abstraction proposed in the modularization plan (Section 7) addresses **partial MMA tile filling** — when the logical data dimension is smaller than the physical MMA tile. This is one important piece of the puzzle, but it's not the whole story. MQA, GQA, and MLA decode differ in more ways than just tile occupancy:

| Variant | KV sharing | Head mapping in decode | What `TileBounds` handles |
|---------|-----------|----------------------|--------------------------|
| **MHA** | No sharing (h_q = h_kv) | Each head is independent → grid/loop dim | N/A (no partial tiles) |
| **MQA** | All Q heads share 1 KV head (h_kv = 1) | All heads share same KV → could pack into M or N | M-masking if packed into M-dim |
| **GQA** | Groups share KV (h_q = g × h_kv) | Group ratio g heads share KV → pack g into N-dim | N-masking if g < N-tile |
| **MLA** | All heads share compressed latent KV | All heads → M-dim | **M-masking when num_heads < 128** |

`TileBounds` cleanly handles the **M-dimension masking** problem that's most acute in MLA decode. But a fully unified decode kernel would also need:

1. **Head mapping strategy** — how heads map to MMA dimensions:
   - MLA: all heads → M-dimension (because KV is shared and latent-compressed)
   - GQA: group ratio → N-dimension (because each group has distinct KV)
   - MQA: similar to GQA with group_size = num_q_heads

2. **KV access pattern** — how KV is loaded:
   - MLA: paged latent + RoPE, decomposed into two GEMM paths
   - GQA/MQA: standard paged or contiguous KV per head group
   - MHA: per-head KV

3. **Softmax scope** — what constitutes a "row" in online softmax:
   - MLA: each head is a row, all rows share the same KV scores
   - GQA: each Q-head within a group has its own row, shares KV scores with sibling heads
   - The row-max/row-sum reductions are identical in algorithm, just differ in which dimension they reduce

The `TileBounds` abstraction handles concern (1)'s **boundary effects** regardless of the mapping strategy. Combined with the modular design where the head mapping strategy is configurable (in `AttentionConfig`), the softmax and epilogue modules can be reused across all variants:

```python
class SoftmaxWarpGroup:
    def step(self, ...):
        rS = tmem_load(tmem_S)

        # Head-dimension masking (compile-time eliminated when not needed)
        if cutlass.const_expr(self.config.tile_bounds.needs_m_masking(tile_m)):
            for i in range(size(rS)):
                if not elem_less(coord_m[i], self.config.tile_bounds.m_bound):
                    rS[i] = -inf

        # KV-dimension masking (causal, sliding window, residual)
        if cutlass.const_expr(self.fusion.mask is not None):
            rS = self.fusion.mask.apply(rS, kv_tile_idx, ...)
```

**Bottom line**: `TileBounds` is necessary but not sufficient for a fully unified MQA/GQA/MLA decode kernel. The complete solution requires `TileBounds` (partial tile masking) + a configurable head mapping strategy (how heads map to MMA dims) + flexible KV access patterns (paged, contiguous, or latent-decomposed). All three are addressed by the modular design's `AttentionConfig`, role modules, and fusion abstractions working together. The key insight is that the *softmax algorithm itself* is identical across all variants — only the data layout feeding into it differs.

---

## 6. Performance Optimizations to Port

Prioritized by expected impact:

### High Priority

1. **Skip-correction optimization** (from DKG decode)
   - When `exp2(scale * (old_max - new_max)) ≈ 1.0`, skip O rescaling entirely
   - Use `vote_all_sync` to check across the warp
   - High impact for decode where correction is often unnecessary

2. **Softmax software pipelining** (from C++ CUTLASS)
   - Interleave FMA, exp2, and dtype conversion with depths 8/16
   - Hides transcendental op latency on prefill's compute-heavy path

### Medium Priority

3. **exp2 emulation** (from C++ CUTLASS)
   - Polynomial approximation faster than hardware `exp2` on SM100
   - Unnecessary on SM103+, so needs architecture dispatch

4. **Fused atomic reduction for split-KV** (from C++ CUTLASS)
   - Eliminates the reduction kernel via `atomicMax` for LSE + `TMA_REDUCE_ADD` for output
   - Saves kernel launch overhead and global memory round-trip

5. **FP8 support** (from both C++ and DKG)
   - E4M3/E5M2 inputs with separate scale_q/k/v/inv_scale_o
   - Requires wider QK MMA tiles (2x K-dim for FP8)

### Lower Priority

6. **Causal-aware tile scheduling** (from C++ CUTLASS)
   - Swizzled launch order: longest mainloop tiles first for load balancing
   - 16x8 super-tile swizzling for L2 cache locality

7. **ThreadShape configurability** (from C++ CUTLASS)
   - `Shape<_2,_1,_1>` vs `Shape<_1,_2,_1>` for how softmax warp-groups partition Q vs K dimensions

8. **LSE output** (from C++ and DKG)
   - Optional log-sum-exp output per row: `lse = log(row_sum) + scale * row_max`

---

## 7. Modularization Plan

### Current State

`prefill.py` (2934 lines) and `mla.py` (3641 lines) are monolithic classes where warp roles, pipeline topology, TMEM layout, softmax algorithm, masking, and scheduling are intertwined. This makes it hard to:
- Add new attention variants without touching the whole kernel
- Reuse softmax logic between prefill and decode
- Experiment with pipeline topologies or TMEM layouts
- Support new architectures without forking

### Target Module Structure

```
flashinfer/cute_dsl/
├── attention/
│   ├── config.py              # AttentionConfig: problem shape, tiles, dtypes
│   ├── tmem_layout.py         # TmemLayout: computed TMEM allocation plan
│   ├── pipeline_topology.py   # PipelineTopology: declarative pipeline graph
│   │
│   ├── roles/                 # One module per warp role
│   │   ├── loader_tma.py      # TMA load warp (Q, K, V)
│   │   ├── loader_cpasync.py  # cp.async load (for paged decode, future)
│   │   ├── loader_pt.py       # Page table load warp
│   │   ├── mma.py             # MMA warp (QK + PV GEMMs)
│   │   ├── softmax.py         # Softmax warp-group
│   │   ├── correction.py      # Correction warp-group
│   │   └── epilogue.py        # Epilogue warp (TMA store)
│   │
│   ├── fusion/                # Attention variant customization
│   │   ├── mask.py            # NoMask, CausalMask, SlidingWindowMask
│   │   ├── logits_transform.py
│   │   ├── output_transform.py
│   │   └── softmax_modifier.py  # Standard, WithSink
│   │
│   ├── scheduler/             # Tile scheduling strategies
│   │   ├── individual.py
│   │   ├── persistent.py
│   │   └── causal_persistent.py
│   │
│   ├── kernels/               # Composed kernel configurations
│   │   ├── prefill.py         # Assembles roles for MHA prefill
│   │   ├── decode.py          # MHA decode
│   │   ├── mla_decode.py      # MLA decode
│   │   └── mla_prefill.py     # MLA prefill
│   │
│   └── wrappers/              # PyTorch-facing API
│       ├── batch_prefill.py
│       └── batch_mla_decode.py
```

### Key Abstractions

#### `AttentionConfig` — Single source of truth

Replaces scattered `self.xxx` attributes. Computed once, passed to all modules.

```python
@dataclass
class AttentionConfig:
    # Problem shape
    head_dim_qk: int
    head_dim_vo: int
    num_heads: int             # Number of attention heads
    num_kv_heads: int = 0      # KV heads (0 = same as num_heads; for GQA/MQA)
    latent_dim: int = 0        # MLA (0 = standard MHA)
    rope_dim: int = 0          # MLA

    # Head mapping (see Section 5)
    head_mapping: HeadMapping = HeadMapping.GRID  # GRID (prefill) or MMA_M (decode)

    # Tile shape
    mma_tiler_mnk: Tuple[int, int, int]
    num_q_tiles_per_cta: int = 2

    # Types
    q_dtype: Type[cutlass.Numeric]
    kv_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32

    # Execution mode
    is_persistent: bool = False
    cluster_shape: Tuple[int, int, int] = (1, 1, 1)
    use_2cta_instrs: bool = False

    # Features
    use_paged_kv: bool = False
    use_split_kv: bool = False

    @cached_property
    def tile_bounds(self) -> TileBounds:
        """Derive tile bounds from head mapping and head count (see Section 5)."""
        if self.head_mapping == HeadMapping.MMA_M:
            return TileBounds(m_bound=self.num_heads)
        return TileBounds()  # No partial-tile masking needed

    @cached_property
    def tmem_layout(self) -> TmemLayout: ...

    @cached_property
    def warp_assignment(self) -> WarpAssignment: ...

    @cached_property
    def pipeline_topology(self) -> PipelineTopology: ...
```

#### `TmemLayout` — Computed (not hardcoded) TMEM map

Derives offsets from tile configuration. Eliminates magic numbers.

```python
class TmemLayout:
    @staticmethod
    def from_config(config):
        # S0@0, S1@tile_m, O0@2*tile_m, O1@3*tile_m
        # P0 aliased inside S1+32, P1 inside S0+32
        # Vec buffers at start of each S region
        ...
```

#### `HeadMapping` — How heads map to MMA dimensions (see Section 5)

```python
class HeadMapping(Enum):
    GRID = "grid"      # Heads in grid/loop dim (prefill, training)
    MMA_M = "mma_m"    # All heads packed into MMA M-dim (MLA decode)
    MMA_N = "mma_n"    # GQA group packed into MMA N-dim (standard GQA decode)
```

This determines the kernel's Q layout and whether `TileBounds` needs M-dimension or N-dimension masking.

#### `TileBounds` — Partial MMA tile masking (see Section 5)

Handles the case where the logical data dimension is smaller than the physical MMA tile. Most critical for MLA decode where `num_heads < 128`.

```python
@dataclass
class TileBounds:
    """Handles the case where the actual problem is smaller than the MMA tile."""
    m_bound: int | None = None  # num_heads for MLA decode, None for prefill
    n_bound: int | None = None  # actual seq_len for residual KV tiles

    def needs_m_masking(self, tile_m: int) -> bool:
        return self.m_bound is not None and self.m_bound < tile_m

    def needs_n_masking(self, tile_n: int) -> bool:
        return self.n_bound is not None and self.n_bound < tile_n
```

Consumed by `SoftmaxWarpGroup` and epilogue modules — compiles away via `cutlass.const_expr` when no masking is needed (e.g., prefill, or MLA decode with exactly 128 heads).

#### `PipelineTopology` — Declarative pipeline graph

```python
@dataclass
class PipelineSpec:
    name: str
    pipeline_type: type    # PipelineTmaUmma, PipelineUmmaAsync, PipelineAsync
    stages: int
    producer_role: WarpRole
    consumer_role: WarpRole
    tx_count: int = 0
```

Enables visualization, validation (no cycles/orphans), and configuration swapping.

#### `AttentionFusion` — Customization bundle

```python
@dataclass
class AttentionFusion:
    mask: Mask = NoMask()
    logits_transform: Callable | None = None
    output_transform: Callable | None = None
    softmax_modifier: SoftmaxModifier | None = None
    # Future: kv_transform, score_mod, block_mask, ...
```

Each option resolves at JIT time via `cutlass.const_expr` — zero overhead when unused.

#### MLA as a config variant, not a fork

```python
# MLA decode with 128 heads (DeepSeek-V3)
mla_128h = AttentionConfig(
    head_dim_qk=576, head_dim_vo=512,
    num_heads=128, latent_dim=512, rope_dim=64,
    head_mapping=HeadMapping.MMA_M,
    num_q_tiles_per_cta=1,  # decode: single Q
    use_paged_kv=True, use_split_kv=True,
    cluster_shape=(2, 1, 1), use_2cta_instrs=True,
)
# tile_bounds.m_bound = 128, no masking needed (128 == tile_m)

# MLA decode with 64 heads (smaller variant)
mla_64h = AttentionConfig(
    head_dim_qk=576, head_dim_vo=512,
    num_heads=64, latent_dim=512, rope_dim=64,
    head_mapping=HeadMapping.MMA_M,
    num_q_tiles_per_cta=1,
    use_paged_kv=True, use_split_kv=True,
    cluster_shape=(1, 1, 1), use_2cta_instrs=False,  # single CTA for 64 heads
)
# tile_bounds.m_bound = 64, masking rows 64-127 in softmax and epilogue

# GQA decode (standard, non-MLA)
gqa_decode = AttentionConfig(
    head_dim_qk=128, head_dim_vo=128,
    num_heads=32, num_kv_heads=8,  # 4:1 GQA ratio
    head_mapping=HeadMapping.MMA_N,
    num_q_tiles_per_cta=1,
    use_paged_kv=True, use_split_kv=True,
)
# GQA group (4 heads) packed into MMA N-dim

# Reuses SoftmaxWarpGroup, CorrectionWarpGroup, etc.
# Only loader and MMA are specialized per variant.
```

---

## 8. Migration Strategy

Each step is independently valuable and testable against the existing monolithic implementation.

### Phase 1: Extract Configuration (Low Risk)

1. **Extract `AttentionConfig`** from scattered `self.xxx` attributes
   - Immediate readability win
   - Single source of truth for all tile sizes, dtypes, feature flags

2. **Extract `TmemLayout`** with computed offsets
   - Eliminates magic numbers (0, 128, 256, 384, 32, 160)
   - Enables auto-derivation for new tile sizes or head dimensions

### Phase 2: Extract Reusable Roles (Medium Risk)

3. **Extract `SoftmaxWarpGroup`** as a standalone class
   - Most reusable piece: identical algorithm in prefill, decode, and MLA
   - Takes `AttentionFusion` for mask/logits_transform/sink hooks

4. **Extract `CorrectionWarpGroup`** as a standalone class
   - Reusable between prefill and decode
   - Takes `AttentionFusion` for output_transform hooks

5. **Bundle `AttentionFusion`** to cleanly separate customization from plumbing
   - Pre-built variants: `STANDARD`, `CAUSAL`, `SIGMOID`, `SINK_CAUSAL`

### Phase 3: Extract Infrastructure (Medium Risk)

6. **Extract `LoaderTMA`** as a base class; specialize `MLALoaderTMA`
   - Enables MLA to share softmax/correction with standard MHA

7. **Extract `PipelineTopology`** as a declarative pipeline graph
   - Enables experimentation with different topologies
   - Simplifies pipeline initialization code

8. **Extract `WarpAssignment`** with named presets
   - `prefill_standard()`, `decode_mla()`, etc.

### Phase 4: Performance Enhancements

9. **Add skip-correction** to `CorrectionWarpGroup`
10. **Add softmax software pipelining** to `SoftmaxWarpGroup`
11. **Add fused atomic reduction** as an alternative to the reduction kernel
12. **Add FP8 support** to config, loader, and MMA modules

### Phase 5: New Features

13. **Causal-aware tile scheduling** with swizzled launch order
14. **ThreadShape configurability** for different Q/K aspect ratios
15. **Backward pass** (requires new kernel composition)

---

## 9. Implementation Status

This section tracks the current state of the modularization effort. The original monoliths (`prefill.py` at 2933 lines, `mla.py` at 3641 lines) remain untouched. A parallel modular implementation lives in `flashinfer/cute_dsl/attention/` (35 files, ~8140 lines total) and is verified by dedicated test suites.

### Current File Layout

```
flashinfer/cute_dsl/attention/          # 8140 lines total across 35 files
│
│  ── Kernels (top-level, readable dispatchers) ──
├── prefill.py              (1230 lines)  FMHA prefill kernel
├── mla_decode.py           (1554 lines)  MLA decode kernel + reduction kernel
│
│  ── Configuration ──
├── config.py               (134 lines)   AttentionConfig, AttentionFusion, HeadMapping
├── mla_config.py            (74 lines)   MLAConfig (separate concrete type)
├── tmem_layout.py           (49 lines)   TmemLayout: computed TMEM offsets
├── warp_schedule.py         (90 lines)   WarpSchedule for FMHA
├── mla_warp_schedule.py     (79 lines)   MLAWarpSchedule (separate concrete type)
├── mainloop_spec.py        (173 lines)   MainloopSpec (FMHA) + MLAMainloopSpec
├── pipeline_topology.py    (315 lines)   PipelineTopology, PipelineEdge, PipelineType, factory
│
│  ── FMHA Roles ──
├── roles/
│   ├── softmax.py          (501 lines)   SoftmaxRole: online softmax with masking
│   ├── correction.py       (458 lines)   CorrectionRole: rescale + epilog
│   ├── mma.py              (258 lines)   MmaRole: QK + PV GEMMs
│   ├── loader_tma.py       (316 lines)   LoaderRole: TMA loads for Q, K, V
│   ├── epilogue.py         (163 lines)   EpilogueRole: TMA store output
│
│  ── MLA Roles ──
│   ├── mla_loader.py       (312 lines)   MLALoaderRole: paged latent + RoPE loads
│   ├── mla_mma.py          (266 lines)   MLAMmaRole: QK + PV with head packing
│   ├── mla_compute.py      (106 lines)   MLAComputeRole: orchestrator
│   ├── mla_softmax.py      (234 lines)   MLASoftmaxRole: online softmax
│   ├── mla_rescale.py       (75 lines)   MLARescaleRole: O accumulator rescaling
│   ├── mla_epilogue.py     (153 lines)   MLAEpilogueRole: final output write
│
│  ── Shared Utilities ──
│   ├── softmax_math.py      (40 lines)   exp2_scale, packed_row_sum
│   ├── tmem_utils.py       (101 lines)   tmem_load_partition
│
│  ── Fusion / Masking ──
├── fusion/
│   ├── mask.py             (160 lines)   MaskType, apply_mask, trip count helpers
│   ├── logits_transform.py               sigmoid_logits_transform
│   ├── output_transform.py               dumb_output_transform
│   └── softmax_modifier.py               (stub)
│
│  ── Schedulers ──
├── scheduler/
│   ├── persistent.py       (166 lines)   FmhaStaticTileScheduler
│   └── mla_persistent.py   (200 lines)   MLAStaticTileScheduler
│
│  ── PyTorch Wrappers ──
└── wrappers/
    ├── batch_prefill.py    (381 lines)   BatchPrefillCuteDSLWrapper
    └── batch_mla.py        (428 lines)   BatchMLAPagedAttentionWrapperCuteDSL
```

### Shared vs Variant-Specific Components

| Component | Shared | FMHA-specific | MLA-specific |
|-----------|--------|---------------|--------------|
| **Pipeline infrastructure** | `PipelineTopology`, `PipelineEdge`, `PipelineType`, `create_pipelines()` | `make_fmha_topology()` | `make_mla_topology()`, `ASYNC_UMMA` type |
| **Mainloop spec** | Dataclass pattern, stage count resolution | `MainloopSpec` | `MLAMainloopSpec` |
| **Warp schedule** | Concept (dataclass with role IDs, register budgets) | `WarpSchedule` (16 warps) | `MLAWarpSchedule` (6-8 warps) |
| **Softmax math** | `exp2_scale()`, `packed_row_sum()` | Used by `SoftmaxRole` | Used by `MLASoftmaxRole` |
| **TMEM utilities** | `tmem_load_partition()` | — | Used by `MLARescaleRole`, `MLAEpilogueRole` |
| **Masking** | `MaskType`, `apply_mask()`, trip count helpers | Used inline in `SoftmaxRole` | Boundary masking inline in `MLASoftmaxRole` |
| **Loader** | — | `LoaderRole` (streaming Q/K/V) | `MLALoaderRole` (paged latent + RoPE) |
| **MMA** | — | `MmaRole` (double-buffered QK/PV) | `MLAMmaRole` (staged, head-packed) |
| **Correction / Rescale** | Packed-scale pattern (similar but not yet extracted) | `CorrectionRole` | `MLARescaleRole` |
| **Epilogue** | — | `EpilogueRole` (TMA store) | `MLAEpilogueRole` (direct write) |
| **Scheduler** | — | `FmhaStaticTileScheduler` | `MLAStaticTileScheduler` |
| **Fusion hooks** | `AttentionFusion` (logits/output transforms, sinks) | Full support | Not yet wired |

### Comparison to C++ CUTLASS Collectives

| Aspect | C++ CUTLASS | Current Python Implementation |
|--------|-------------|-------------------------------|
| **Kernel template** | One shared kernel (`Sm100FmhaFwdKernelTmaWarpspecialized`) parameterized by mainloop type | Separate kernel files (`prefill.py`, `mla_decode.py`) — both are thin dispatchers |
| **Mainloop** | `Sm100FmhaFwd*` / `Sm100FmhaMlaFwd*` concrete types defining pipelines + TMEM + roles | `MainloopSpec` / `MLAMainloopSpec` dataclasses with `PipelineTopology` + `TmemLayout` |
| **Pipeline creation** | Types defined in mainloop, instantiated by kernel | Declarative `PipelineTopology` with `create_pipelines()` factory |
| **Roles** | Methods on the mainloop (`load()`, `mma()`, `softmax()`, `correction()`) | Separate role classes composed by the kernel |
| **Shared math** | Inline in each mainloop (no cross-variant sharing) | Extracted: `softmax_math.py`, `tmem_utils.py` |
| **Config** | Template parameters + `Params` struct | `AttentionConfig` / `MLAConfig` dataclasses |
| **CollectiveBuilder** | Selects MMA atoms, TMA descriptors, pipeline types from config | Not yet implemented |

The Python implementation actually shares *more* code across variants than C++ CUTLASS, which keeps each mainloop self-contained. The tradeoff is that C++ gets maximum compile-time optimization per variant while Python uses JIT compilation that naturally specializes per call.

### Test Status

| Test Suite | Total | Pass | Fail | Notes |
|------------|-------|------|------|-------|
| `test_blackwell_fmha_attention.py` | 252 | 252 | 0 | Full coverage: causal, sliding window, GQA, sinks |
| `test_blackwell_mla_attention.py` | 24 | 14 | 10 | Failures are pre-existing upstream bugs (same in original `mla.py`) |

### Remaining Work

Listed roughly in order of impact:

1. **Extract `packed_scale` utility** — The `mul_packed_f32x2` loop pattern appears in both `CorrectionRole.rescale()` and `MLARescaleRole.run()`. ~5 lines each. Low risk.

2. **Move role instantiation into MainloopSpec** — Currently the kernel `__init__` creates role instances. Moving this into the mainloop spec would make it closer to the C++ pattern where the mainloop *is* the collective. Medium risk.

3. **Add a CollectiveBuilder** — A factory that selects MMA atoms, TMA descriptors, and pipeline types based on config. Would further reduce kernel boilerplate. Medium risk.

4. **Wire `AttentionFusion` into MLA** — The fusion hooks (logits transforms, output transforms, attention sinks) currently only work in FMHA. Extending to MLA would enable customizable MLA decode. Low-medium risk.

5. **Performance optimizations** — Skip-correction (`vote_all_sync` to avoid rescaling when unnecessary), softmax software pipelining, exp2 emulation, fused atomic reduction for split-KV. These are independent and can be added per-variant.

6. **Unify kernel dispatcher** — The C++ code uses one kernel template for both FMHA and MLA. Currently we have two separate kernel files. Unifying would require abstracting the warp dispatch, which is the most variant-specific part. High risk, potentially not worth it given Python's JIT advantage.

---

## Appendix: File Reference

### FlashInfer PR #1549

| File | Lines | Description |
|------|-------|-------------|
| `flashinfer/cute_dsl/prefill.py` | 2934 | MHA prefill kernel |
| `flashinfer/cute_dsl/mla.py` | 3641 | MLA decode kernel |
| `flashinfer/cute_dsl/patch/pipeline.py` | 419 | Pipeline producer/consumer wrappers |
| `tests/test_blackwell_fmha.py` | 570+ changed | Prefill tests (causal, varlen, GQA, sink) |
| `tests/test_deepseek_mla.py` | 119+ changed | MLA decode tests |
| `tests/sink_attention_reference.py` | 403 | Sink attention reference implementation |

### DKG Python CuTe DSL

| File | Description |
|------|-------------|
| `fmha.py` | MHA prefill (~3620 lines) |
| `fmha_decode.py` | MHA decode with split-KV |
| `fmha_bwd.py` | Backward pass |
| `mla/mla_decode_fp16.py` | MLA decode FP16 |
| `mla/mla_decode_fp8.py` | MLA decode FP8 |
| `mla/mla_helpers.py` | MLA shared infrastructure |
| `mixed_input_fmha/` | Mixed-precision variants (Int8/Int4/FP8 K/V) |

### C++ CUTLASS

| File | Description |
|------|-------------|
| `device/fmha.hpp` | Device-level launch wrapper |
| `kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp` | Forward kernel entry |
| `collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp` | Forward mainloop (mma, softmax, correction) |
| `collective/sm100_fmha_load_tma_warpspecialized.hpp` | TMA loader |
| `collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp` | Epilogue (TMA store) |
| `collective/fmha_common.hpp` | Masks, variable-length, GEMM helpers |
| `kernel/sm100_fmha_gen_kernel_warpspecialized.hpp` | Gen/decode kernel |
| `kernel/sm100_fmha_mla_tma_warpspecialized.hpp` | MLA inference kernel |
| `kernel/sm100_fmha_mla_reduction.hpp` | MLA split-KV reduction |
