---
name: Refactor CuTe DSL attention
overview: Create `flashinfer/cute_dsl/attention/` with kernels at the top level as readable compositions of lower-level building blocks (config, roles, fusion, scheduler), populated with the prefill kernel, verified with working tests.
todos:
  - id: create-dirs
    content: "Create the full directory structure: attention/, attention/roles/, attention/fusion/, attention/scheduler/, attention/wrappers/"
    status: in_progress
  - id: config
    content: Create config.py with AttentionConfig, HeadMapping, TileBounds, AttentionFusion
    status: pending
  - id: tmem-layout
    content: Create tmem_layout.py with TmemLayout computed from config
    status: pending
  - id: fusion-mask
    content: Create fusion/mask.py with MaskType enum
    status: pending
  - id: fusion-transforms
    content: Create fusion/logits_transform.py and fusion/output_transform.py with example callbacks
    status: pending
  - id: scheduler
    content: Create scheduler/persistent.py with FmhaStaticTileScheduler (lines 111-262)
    status: pending
  - id: roles-stubs
    content: Create roles/ stubs for softmax.py, correction.py, mma.py, loader_tma.py, epilogue.py
    status: pending
  - id: pipeline-stub
    content: Create pipeline_topology.py stub with PipelineSpec dataclass
    status: pending
  - id: prefill-kernel
    content: Create top-level prefill.py — the kernel composing building blocks
    status: pending
  - id: wrapper
    content: Create wrappers/batch_prefill.py with BatchPrefillCuteDSLWrapper
    status: pending
  - id: init-files
    content: Create __init__.py files at each level with re-exports
    status: pending
  - id: backward-compat
    content: Replace original prefill.py with re-exports from attention/
    status: pending
  - id: verify-tests
    content: Run existing test_blackwell_fmha.py to verify everything works
    status: pending
isProject: false
---

# Refactor CuTe DSL Attention — Kernels as Top-Level Compositions

## Design Principle

**Kernels live at the top level** of the `attention/` package. They are the readable, high-level compositions that express the core mathematical algorithm. Building blocks (config, roles, fusion, scheduler, pipeline) live one level below in subdirectories. When you open `attention/prefill.py`, you should see something close to:

> "Load Q, K, V tiles. Compute S = QK^T. Apply mask. Softmax. Compute O = PV. Correct. Write output."

The building blocks handle the *how* (TMEM layout, pipeline synchronization, warp assignment). The kernel expresses the *what*.

## Target Directory Structure

```
flashinfer/cute_dsl/attention/
├── __init__.py                    # Re-exports public API
│
│  ── Kernels (top-level: readable compositions) ──
├── prefill.py                     # MHA prefill kernel (composes building blocks)
├── decode.py                      # (future) MHA decode kernel
├── mla_decode.py                  # (future) MLA decode kernel
│
│  ── Building blocks (one level below) ──
├── config.py                      # AttentionConfig, HeadMapping, TileBounds, AttentionFusion
├── tmem_layout.py                 # TmemLayout: computed TMEM offsets
├── pipeline_topology.py           # (stub) PipelineTopology, PipelineSpec
│
├── roles/                         # Warp role modules
│   ├── __init__.py
│   ├── softmax.py                 # (stub) SoftmaxWarpGroup
│   ├── correction.py              # (stub) CorrectionWarpGroup
│   ├── mma.py                     # (stub) MMA warp
│   ├── loader_tma.py              # (stub) TMA loader
│   └── epilogue.py                # (stub) Epilogue warp
│
├── fusion/                        # Attention variant customization
│   ├── __init__.py
│   ├── mask.py                    # MaskType enum
│   ├── logits_transform.py        # sigmoid_logits_transform, etc.
│   ├── output_transform.py        # dumb_output_transform, etc.
│   └── softmax_modifier.py        # (stub) WithSink modifier
│
├── scheduler/                     # Tile scheduling strategies
│   ├── __init__.py
│   └── persistent.py              # FmhaStaticTileScheduler + params
│
└── wrappers/                      # PyTorch-facing API
    ├── __init__.py
    └── batch_prefill.py           # BatchPrefillCuteDSLWrapper + tensor helpers
```

## What Gets Built Now

### `config.py` — Single source of truth

Extract from `BlackwellFusedMultiHeadAttentionForward.__init__` (lines 316-395 of current `prefill.py`):

```python
@dataclass
class AttentionConfig:
    # Core parameters (from __init__ args)
    qk_acc_dtype: Type[cutlass.Numeric]
    pv_acc_dtype: Type[cutlass.Numeric]
    mma_tiler: Tuple[int, int, int]
    is_persistent: bool
    mask_type: MaskType
    num_repeat_kv_heads: int = 1
    window_left: int = -1

    # Derived (computed in __init__, currently lines 318-329)
    @cached_property
    def cta_tiler(self): ...        # (2*M, N, K)
    @cached_property
    def qk_mma_tiler(self): ...     # same as mma_tiler
    @cached_property
    def pv_mma_tiler(self): ...     # (M, K, N) transposed

    # Future extensions
    head_mapping: HeadMapping = HeadMapping.GRID
    num_heads: int = 0
    num_kv_heads: int = 0

@dataclass
class AttentionFusion:
    logits_transform: Callable | None = None
    output_transform: Callable | None = None
    M_D_update: Callable | None = None
    use_attention_sink: bool = False
    custom_params: SimpleNamespace | None = None
```

### `tmem_layout.py` — Computed offsets

Extract hardcoded magic numbers from lines 358-372:

```python
@dataclass
class TmemLayout:
    s0_offset: int      # Score buffer 0 (currently 0)
    s1_offset: int      # Score buffer 1 (currently 128)
    o0_offset: int      # Output buffer 0 (currently 256)
    o1_offset: int      # Output buffer 1 (currently 384)
    p0_offset: int      # P buffer 0 (currently 32)
    p1_offset: int      # P buffer 1 (currently 160)
    vec0_offset: int    # Vec buffer 0 for row_max/sum (currently 0)
    vec1_offset: int    # Vec buffer 1 for row_max/sum (currently 128)

    @staticmethod
    def from_config(config: AttentionConfig) -> "TmemLayout":
        tile_m = config.mma_tiler[0]
        return TmemLayout(
            s0_offset=0, s1_offset=tile_m,
            o0_offset=2*tile_m, o1_offset=3*tile_m,
            p0_offset=tile_m//4, p1_offset=tile_m + tile_m//4,
            vec0_offset=0, vec1_offset=tile_m,
        )
```

### `fusion/mask.py`

Move `MaskType` enum (lines 264-268):

```python
class MaskType(enum.Enum):
    NO_MASK = enum.auto()
    RESIDUAL_MASK = enum.auto()
    CAUSAL_MASK = enum.auto()
    SLIDING_WINDOW_MASK = enum.auto()
```

### `fusion/logits_transform.py` and `fusion/output_transform.py`

Move example callbacks (lines 2574-2581).

### `scheduler/persistent.py`

Move `FmhaStaticTileScheduler`, `FmhaStaticTileSchedulerParams`, and factory functions (lines 111-262). Self-contained, only depends on `cutlass.cute`.

### `prefill.py` — The top-level kernel

This is the heart of the refactoring. `BlackwellFusedMultiHeadAttentionForward` (~2300 lines) moves here. The constructor changes to accept `AttentionConfig` and `AttentionFusion`:

```python
from .config import AttentionConfig, AttentionFusion
from .tmem_layout import TmemLayout
from .fusion.mask import MaskType
from .scheduler.persistent import FmhaStaticTileScheduler

class BlackwellFusedMultiHeadAttentionForward:
    def __init__(self, config: AttentionConfig, fusion: AttentionFusion | None = None):
        self.config = config
        self.fusion = fusion or AttentionFusion()
        self.tmem = TmemLayout.from_config(config)

        # Warp assignment (stays inline for now, future extraction to roles/)
        self.softmax0_warp_ids = (0, 1, 2, 3)
        ...
```

Inside the kernel body, attribute references change:

- `self.qk_acc_dtype` -> `self.config.qk_acc_dtype`
- `self.tmem_s0_offset` -> `self.tmem.s0_offset`
- `self.mask_type` -> `self.config.mask_type`
- `self.logits_transform` -> `self.fusion.logits_transform`
- `self.custom_logits_transform` -> `(self.fusion.logits_transform is not None)`

The `__call__` method and all internal warp role methods stay functionally identical.

### `wrappers/batch_prefill.py` — PyTorch-facing API

`BatchPrefillCuteDSLWrapper` + helper functions. Its `plan()` method changes to construct `AttentionConfig` + `AttentionFusion` and pass them to the kernel:

```python
config = AttentionConfig(
    qk_acc_dtype=self._qk_acc_dtype,
    pv_acc_dtype=self._pv_acc_dtype,
    mma_tiler=self._mma_tiler,
    is_persistent=self._is_persistent,
    mask_type=self._mask_type,
    num_repeat_kv_heads=h_r,
    window_left=window_left,
)
fusion = AttentionFusion(
    logits_transform=logits_transform,
    output_transform=output_transform,
    M_D_update=M_D_update,
    use_attention_sink=use_attention_sink,
    custom_params=custom_params,
)
fmha = BlackwellFusedMultiHeadAttentionForward(config, fusion)
```

### Stub files

Each stub file in `roles/` and `pipeline_topology.py` contains:

- Module docstring explaining what will be extracted and from where
- The target class signature (no implementation)
- A reference to the line ranges in the current `prefill.py` where the code lives

Example `roles/softmax.py` stub:

```python
"""SoftmaxWarpGroup — to be extracted from prefill.py softmax_step() methods.

Handles online softmax computation including:
- Row-max tracking and exp2 computation
- Row-sum accumulation
- KV-dimension masking (causal, sliding window)
- Head-dimension masking via TileBounds (for MLA decode)
- Logits transform hooks via AttentionFusion

Source: BlackwellFusedMultiHeadAttentionForward.softmax_step() in prefill.py
"""
```

## Backward Compatibility

`[flashinfer/cute_dsl/prefill.py](flashinfer/cute_dsl/prefill.py)` becomes a re-export shim:

```python
from .attention.prefill import BlackwellFusedMultiHeadAttentionForward
from .attention.wrappers.batch_prefill import (
    BatchPrefillCuteDSLWrapper, qkv_torch_2_cute, create_and_pad_tensor,
)
from .attention.fusion.mask import MaskType
from .attention.scheduler.persistent import (
    FmhaStaticTileScheduler, FmhaStaticTileSchedulerParams,
    create_fmha_static_tile_scheduler, create_fmha_static_tile_scheduler_params,
)
from .attention.fusion.logits_transform import sigmoid_logits_transform
from .attention.fusion.output_transform import dumb_output_transform
```

## Test Verification

Run existing test unchanged (it imports from `flashinfer.cute_dsl.prefill` which re-exports):

```
pytest tests/test_blackwell_fmha.py::test_blackwell_cutedsl_fmha[dtype0-False-1.0-128-128-8-32-256-1] -v
```

## What This Enables Next

With kernels at the top level and building blocks below:

1. **Extract `SoftmaxWarpGroup**` into `roles/softmax.py` — the prefill kernel's `softmax_step()` becomes a call to `self.softmax.step()`
2. **Extract `CorrectionWarpGroup**` into `roles/correction.py`
3. **Add `mla_decode.py**` at the top level — composes the same `roles/softmax.py` and `roles/correction.py` with MLA-specific loader and config
4. **Fill in `pipeline_topology.py**` — replace the ~80 lines of imperative pipeline setup with a declarative graph
5. Anyone reading `prefill.py` or `mla_decode.py` sees the algorithm; the building blocks handle the mechanics

