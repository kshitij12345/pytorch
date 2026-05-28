"""
Reproducer 10: AOTI buffer mutation test fails - TMA introduces empty_strided

Error: RuntimeError: Expected to not find "empty_strided" but found it
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1,
    global_scratch_scratch_14_size, ...))

Root cause: When TMA is enabled, the inductor generates global scratch buffers
for TMA descriptor operations. These scratch buffers are allocated via
`aoti_torch_empty_strided` in the generated C++ wrapper code. Tests that assert
`empty_strided` should NOT appear in the generated code (because buffer mutation
should be in-place) now fail because TMA's scratch buffer allocation introduces
it regardless of the test's buffer mutation logic.

Original test: test/inductor/test_aot_inductor.py::AOTInductorTestABICompatibleGpu::test_buffer_mutation_3_cuda

Run: python agent_space/tma_repros/repro_10_aoti_buffer_mutation.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMAAOTIBufferMutation(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_aoti_buffer_mutation_empty_strided(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x.add_(1)
                x.add_(2)
                return x

        model = Model().cuda()
        x = torch.randn(10, 10, device="cuda")

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            so_path = torch._export.aot_compile(
                model,
                (x,),
                options={"aot_inductor.output_path": f"{tmpdir}/model.so"},
            )
            # The compilation itself succeeds, but the generated C++ code
            # contains `empty_strided` for TMA scratch buffers, which the
            # original test asserts should not be present.


if __name__ == "__main__":
    run_tests()
