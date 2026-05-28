"""
Reproducer 6: Triton runtime allocator not set

Error: Exception: Kernel requires a runtime memory allocation, but no
allocator was set. Use triton.set_allocator to specify an allocator.

Root cause: TMA descriptor operations in certain kernels (those using
operations like torch.complex or torch.kthvalue) require Triton's runtime
memory allocator for scratch space. When use_tensor_descriptor is enabled
globally, some kernels that previously used plain loads now use TMA
descriptors which internally need runtime allocation, but PyTorch's Triton
integration hasn't configured the allocator for this path.

NOTE: This error was observed on H100 (sm_90) in CI. It may not reproduce
on Blackwell (sm_100) where TMA implementation details differ. The test
below documents the exact invocation from CI.

Original test: test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCUDA::test_comprehensive_complex_cuda_float64
Also: test/inductor/test_torchinductor_opinfo.py::TestInductorOpInfoCUDA::test_comprehensive_kthvalue_cuda_float16

To reproduce from CI:
    PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=7 python test/inductor/test_torchinductor_opinfo.py TestInductorOpInfoCUDA.test_comprehensive_complex_cuda_float64

Run: python agent_space/tma_repros/repro_06_triton_allocator_not_set.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMATritonAllocatorNotSet(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_sort_allocator_error(self):
        """
        On H100 (sm_90), certain TMA descriptor operations require Triton's
        runtime memory allocator. This test demonstrates the issue with sort
        operations that internally use TMA descriptors with scratch space.

        The error "Kernel requires a runtime memory allocation, but no allocator
        was set" occurs on H100 but may not reproduce on Blackwell (sm_100)
        where the TMA implementation differs.
        """

        def fn(x):
            return torch.sort(x, dim=-1, stable=True)

        x = torch.randn(5, 5, 5, device="cuda", dtype=torch.float16)

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        actual = compiled_fn(x)
        torch.testing.assert_close(actual.values, expected.values)
        torch.testing.assert_close(actual.indices, expected.indices)


if __name__ == "__main__":
    run_tests()
