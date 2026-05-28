"""
Reproducer 7: Out of shared memory with TMA on large permuted reductions

Error: RuntimeError: No valid triton configs. OutOfResources: out of resource:
shared memory, Required: 4194312, Hardware limit: 232448.

Root cause: TMA descriptors consume additional shared memory for descriptor
metadata. When combined with large tiled reductions on permuted (non-contiguous)
tensors, the total shared memory required exceeds hardware limits. The kernel
that previously fit within shared memory now fails because TMA overhead pushes
it over the 232KB hardware limit on H100.

NOTE: This error was observed on H100 (sm_90) with 232KB shared memory limit.
On Blackwell (sm_100) with larger shared memory, this test passes. The test
documents the exact pattern that triggers the issue.

Original test: test/inductor/test_torchinductor.py::TritonCodeGenTests::test_evict_last_non_coalesced_loads

To reproduce on H100:
    python test/inductor/test_torchinductor.py TritonCodeGenTests.test_evict_last_non_coalesced_loads
    (with conftest.py enabling use_tensor_descriptor=True, assume_aligned_inputs=True)

Run: python agent_space/tma_repros/repro_07_shared_memory_overflow.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMASharedMemoryOverflow(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_permuted_reduction_shared_memory_overflow(self):
        """
        512^3 permuted tensors with a reduction create a kernel that requires
        ~4MB of shared memory with TMA enabled. H100 has only 232KB limit,
        causing OutOfResources. Blackwell has more smem so this may pass there.
        """

        @torch.compile
        def f(a, b):
            return (a * b).sum(dim=-1)

        N = 512
        inps = (
            torch.randn(N, N, N, device="cuda").permute(2, 1, 0),
            torch.randn(N, N, N, device="cuda").permute(1, 2, 0),
        )
        result = f(*inps)
        expected = (inps[0] * inps[1]).sum(dim=-1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    run_tests()
