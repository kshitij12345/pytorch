"""
Reproducer 4: AOTInductor C++ wrapper codegen error - undeclared variable

Error: 'cached_torch_device_type_cuda' was not declared in this scope

Root cause: When TMA introduces a global_scratch buffer allocation via
aoti_torch_empty_strided, the C++ wrapper codegen uses the cached variable
`cached_torch_device_type_cuda` which hasn't been declared in the function
scope. This is a codegen bug in how scratch buffers are allocated in the
AOTI C++ wrapper when TMA is active.

Original test: test/inductor/test_aot_inductor.py::AOTInductorTestABICompatibleGpu::test_sdpa_2_cuda

Run: python agent_space/tma_repros/repro_04_aoti_compile_error.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMAAOTICompileError(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_aoti_sdpa_scratch_buffer_codegen(self):
        class Model(torch.nn.Module):
            def forward(self, q, k, v, x):
                t = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )[0]
                return x + t

        model = Model().cuda()
        example_inputs = (
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device="cuda"),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device="cuda"),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device="cuda"),
            torch.randn(1, 48, 64, 64, dtype=torch.bfloat16, device="cuda"),
        )

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            so_path = torch._export.aot_compile(
                model,
                example_inputs,
                options={"aot_inductor.output_path": f"{tmpdir}/model.so"},
            )


if __name__ == "__main__":
    run_tests()
