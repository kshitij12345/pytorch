"""
Reproducer 3: CUDA illegal memory access in combo kernel with channels_last

Error: CUDA error: an illegal memory access was encountered

Root cause: Combo kernels with channels_last memory format produce TMA
descriptors whose grid/offset calculations don't correctly account for
the non-contiguous layout, causing out-of-bounds memory accesses.

WARNING: This test corrupts the CUDA context. Run in isolation.

Original test: test/inductor/test_combo_kernels.py::ComboKernelTests::test_combo_kernel_per_config_subkernel_block_size

Run: python agent_space/tma_repros/repro_03_combo_kernel_illegal_access.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMAComboKernelIllegalAccess(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": False,
        }
    )
    def test_combo_kernel_channels_last_oob(self):
        def fn(t0, t1, t2, t3, t4, t5, t6, t7):
            o0 = t0.contiguous(memory_format=torch.channels_last)
            o1 = t1.contiguous(memory_format=torch.channels_last)
            o2 = t2.contiguous(memory_format=torch.channels_last)
            o3 = t3.contiguous(memory_format=torch.channels_last)
            o4 = t4.contiguous(memory_format=torch.channels_last)
            o5 = t5.contiguous(memory_format=torch.channels_last)
            o6 = t6.contiguous(memory_format=torch.channels_last)
            o7 = t7.contiguous(memory_format=torch.channels_last)
            return o0, o1, o2, o3, o4, o5, o6, o7

        inps = [
            torch.randn(4, 3, 224, 224, device="cuda"),
            torch.randn(64, 3, 3, 3, device="cuda"),
            torch.randn(64, 64, 3, 3, device="cuda"),
            torch.randn(128, 64, 3, 3, device="cuda"),
            torch.randn(128, 128, 3, 3, device="cuda"),
            torch.randn(256, 128, 3, 3, device="cuda"),
            torch.randn(256, 256, 3, 3, device="cuda"),
            torch.randn(256, 256, 3, 3, device="cuda"),
        ]

        expected = fn(*inps)
        compiled_fn = torch.compile(fn)
        actual = compiled_fn(*inps)
        for e, a in zip(expected, actual):
            torch.testing.assert_close(a, e)


if __name__ == "__main__":
    run_tests()
