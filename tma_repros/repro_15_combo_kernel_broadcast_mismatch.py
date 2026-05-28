"""
Reproducer 15: Combo kernel TMA broadcast shape mismatch

Error: Cannot broadcast, the expanded size of the tensor (128) must match
the existing size (2048) at non-singleton dimension 1

Root cause: In combo kernels with mixed persistent/regular reductions, the
TMA descriptor loads using the wrong block size variable (R0_BLOCK instead
of R0_BLOCK_0). This causes the loaded tensor to have a different shape than
expected, and the subsequent `tl.broadcast_to` fails because the shapes
don't match (TMA loaded [XBLOCK, R0_BLOCK=2048] but code expects
[XBLOCK, R0_BLOCK_0=128]).

This combines two bugs: (1) the R0_BLOCK naming issue from repro_01, and
(2) when R0_BLOCK resolves to a different value than R0_BLOCK_0, it creates
a broadcast shape mismatch rather than just a NameError.

Original test: test/inductor/test_combo_kernels.py::ComboKernelTests::test_combo_kernel_per_config_subkernel_red_per

Run: python agent_space/tma_repros/repro_15_combo_kernel_broadcast_mismatch.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMAComboKernelBroadcastMismatch(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": False,
        }
    )
    def test_combo_kernel_red_per_broadcast_mismatch(self):
        """Mixed persistent/regular reduction combo kernel where TMA loads
        with wrong block size causing broadcast shape mismatch."""

        def fn(a, b):
            r1 = a.sum(dim=-1)
            r2 = b.sum(dim=-1)
            return r1, r2

        inps = [
            torch.randn(512, 128, device="cuda"),  # Persistent (r0=128)
            torch.randn(256, 2048, device="cuda"),  # Regular (r0=2048)
        ]
        expected = fn(*inps)
        compiled_fn = torch.compile(fn)
        actual = compiled_fn(*inps)
        torch.testing.assert_close(expected[0], actual[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(expected[1], actual[1], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
