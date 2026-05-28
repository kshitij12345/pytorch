# TMA Tensor Descriptor Failure Reproducers

Reproducers for failures when running inductor tests with:
```python
triton.use_tensor_descriptor = True
assume_aligned_inputs = True
```

Pipeline: https://gitlab-master.nvidia.com/dl/pytorch/update-scripts/-/pipelines/52624097

## Known Fixes (already have PRs)

These PRs fix the **bool dtype** and **misaligned buffer** issues which account for ~80% of
total pipeline failures, but do NOT fix any of the reproducers below:

- **PR #185223** — Skips TMA for dtypes without `CUtensorMapDataType` mapping (e.g. `torch.bool` / `tl.int1`)
- **PR #184717** — Refuses TMA for buffers recorded in `V.graph.unaligned_buffers`

---

## Reproducer Summary

All reproducers are independent, self-contained Python files. Run with:
```bash
python tma_repros/repro_XX_name.py
```

### Confirmed failing locally (12 reproducers)

| # | File | Error | Root Cause |
|---|------|-------|------------|
| 01 | `repro_01_combo_kernel_name_error.py` | `NameError('R0_BLOCK is not defined')` | TMA codegen emits `R0_BLOCK` but combo kernel scope only defines `R0_BLOCK_0` |
| 02 | `repro_02_combo_kernel_small_block.py` | `1 * 4 = 4 bytes < 16` | Combo kernel reduction with XBLOCK=1 bypasses the `no_x_dim` store guard |
| 03 | `repro_03_combo_kernel_illegal_access.py` | `CUDA error: illegal memory access` | TMA descriptors + channels_last combo kernels access OOB memory |
| 04 | `repro_04_aoti_compile_error.py` | `'cached_torch_device_type_cuda' undeclared` | AOTI C++ wrapper doesn't declare cached device type var for TMA scratch buffers |
| 05 | `repro_05_quantization_accuracy.py` | Numerical: 8.6% elements wrong | FP8 quantization scaling produces incorrect results with TMA |
| 08 | `repro_08_l1_cache_assertion.py` | `[] is not true` (no `tl.load` found) | TMA replaces all `tl.load` calls, breaking tests that inspect load patterns |
| 09 | `repro_09_aoti_misaligned_clone.py` | Unwanted `clone_preserve_strides` | TMA 16-byte alignment forces copies of element-misaligned inputs |
| 11 | `repro_11_split_scan_xblock_assert.py` | `assert tconfig.kwargs["XBLOCK"] == 1` | Split scan decorator requires XBLOCK=1, TMA autotuning changes it |
| 12 | `repro_12_kernel_fusion_store_count.py` | `Expected 1 but got 0` | TMA descriptor `.store()` replaces `tl.store()`, breaking store-count assertions |
| 13 | `repro_13_aoti_eager_failure.py` | `Failed to produce kernel library` | AOTI eager op compilation fails (same C++ bug as repro_04) |
| 14 | `repro_14_small_normalized_shape.py` | `2 * 4 = 8 bytes < 16` | Persistent reduction store skips the 16-byte block check |
| 15 | `repro_15_combo_kernel_broadcast_mismatch.py` | `Cannot broadcast (128) != (2048)` | R0_BLOCK vs R0_BLOCK_0 mismatch causes shape error in broadcast |

### H100-specific (3 reproducers, do not reproduce on Blackwell)

| # | File | Error | Note |
|---|------|-------|------|
| 06 | `repro_06_triton_allocator_not_set.py` | `no allocator was set` | Triton runtime allocator needed for TMA on H100 |
| 07 | `repro_07_shared_memory_overflow.py` | `OutOfResources: shared memory` | H100 has 232KB smem limit; Blackwell has more |
| 10 | `repro_10_aoti_buffer_mutation.py` | `empty_strided` in codegen | Test assertion issue (compile succeeds) |

---

## Bug Categories

### 1. Combo kernel variable naming (repros 01, 15)

The TMA codegen emits block shape references like `R0_BLOCK` but in combo kernels
the actual variables are suffixed (`R0_BLOCK_0`, `R0_BLOCK_1`). This causes either
a `NameError` (when `R0_BLOCK` doesn't exist) or a broadcast shape mismatch (when
`R0_BLOCK` resolves to a different sub-kernel's value).

**Code location:** TMA descriptor block_shape generation in `torch/_inductor/codegen/triton.py`

### 2. Small block size < 16 bytes (repros 02, 14)

The TMA API requires at least 16 bytes in the innermost block dimension. Two cases
slip through the existing guards:
- **Combo kernels** with XBLOCK=1 bypass the `no_x_dim` check (which only triggers
  when the kernel itself has no x dimension, not when a sub-kernel happens to get XBLOCK=1)
- **Persistent reduction stores** where the innermost block is R0_BLOCK: the 16-byte
  validation only runs for loads (`not self.for_store`), not stores

**Code location:** `TMACompatibilityChecker.can_use_tma()` and `are_block_parameters_compatible()`

### 3. CUDA illegal memory access (repro 03)

Combo kernels with channels_last tensors produce TMA descriptors whose offset
calculations don't account for the non-contiguous stride pattern, causing OOB access.

**WARNING:** This test corrupts the CUDA context. Run in isolation.

### 4. AOTI C++ codegen bugs (repros 04, 09, 13)

When TMA introduces scratch buffers (`global_scratch`), the generated C++ wrapper:
- References `cached_torch_device_type_cuda` which isn't declared in per-kernel function scope
- Inserts `aoti_torch_clone_preserve_strides` for alignment that wasn't needed before
- Fails entirely in AOTI eager mode (same undeclared variable)

**Code location:** `torch/_inductor/codegen/wrapper.py` (AOTI C++ wrapper generation)

### 5. Numerical accuracy (repro 05)

FP8 quantization with TMA produces wrong results (8.6% of elements, up to 0.17 absolute
difference). The scaling + cast to FP8 + cast back path generates a TMA descriptor that
doesn't correctly handle the precision requirements.

### 6. Test infrastructure conflicts (repros 08, 12)

TMA fundamentally changes the generated code patterns:
- `tl.load(ptr, ...)` → `tl.make_tensor_descriptor(ptr, ...).load(...)`
- `tl.store(ptr, ...)` → `tl.make_tensor_descriptor(ptr, ...).store(...)`

Tests that inspect generated code for `tl.load`/`tl.store` patterns get 0 matches.

### 7. Split scan assertion (repro 11)

The `@split_scan` heuristic decorator asserts `XBLOCK == 1`. With TMA enabled,
autotuning may select configs with XBLOCK > 1, violating this invariant.

**Code location:** `torch/_inductor/runtime/triton_heuristics.py:2902`

---

## Running

```bash
# Run a single reproducer
python tma_repros/repro_01_combo_kernel_name_error.py

# Run all (except repro_03 which corrupts CUDA context)
for f in 01 02 04 05 08 09 11 12 13 14 15; do
  echo "=== repro_${f} ==="
  python tma_repros/repro_${f}_*.py 2>&1 | tail -3
done

# Run repro_03 in isolation
python tma_repros/repro_03_combo_kernel_illegal_access.py
```

## Environment

- PyTorch: main branch (commit 339e396)
- Triton: bundled with PyTorch
- GPU: H100 (sm_90) or Blackwell (sm_100)
- CUDA: 12.8+ / 13.3
