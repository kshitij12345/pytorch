"""
Reproducer 1: Combo kernel NameError - R0_BLOCK undefined

Error: NameError('R0_BLOCK is not defined')

Root cause: In combo kernels, reduction block variables are named R0_BLOCK_0,
R0_BLOCK_1, etc. But the TMA codegen emits
`tl.make_tensor_descriptor(..., block_shape=[XBLOCK, R0_BLOCK])` using the
un-suffixed name, which doesn't exist in the combo kernel scope.

Original test: test/inductor/test_combo_kernels.py::ComboKernelTests::test_combo_kernel_per_config_subkernel_per

Run: python agent_space/tma_repros/repro_01_combo_kernel_name_error.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMAComboKernelNameError(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": False,
        }
    )
    def test_combo_kernel_per_subkernel_rblock_name(self):
        def fn(a, b):
            return a.sum(dim=-1), b.sum(dim=-1)

        inps = [
            torch.randn(1024, 64, device="cuda"),
            torch.randn(1024, 512, device="cuda"),
        ]
        expected = fn(*inps)
        compiled_fn = torch.compile(fn)
        actual = compiled_fn(*inps)
        torch.testing.assert_close(expected, actual, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
