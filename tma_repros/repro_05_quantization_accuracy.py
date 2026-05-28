"""
Reproducer 5: TMA numerical accuracy failure in FP8 quantization path

Error: AssertionError: Tensor-likes are not close!
  Mismatched elements: 22 / 256 (8.6%)
  Greatest absolute difference: 0.171875

Root cause: When TMA descriptors are used in FP8 quantization scaling
operations, the generated kernel produces numerically wrong results.
The scaling + cast to FP8 + cast back path generates a TMA descriptor
that doesn't correctly handle the precision requirements of the scaling
computation.

Original test: test/inductor/test_quantization.py::TestQuantization::test_activation_quantization_aten_with_scaling

Run: python agent_space/tma_repros/repro_05_quantization_accuracy.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMAQuantizationDescriptor(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_activation_quantization_tma_failure(self):
        class SimpleQuantModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 16, bias=False)

            def forward(self, x, w):
                scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
                x_scaled = x / scale
                x_fp8 = x_scaled.to(torch.float8_e4m3fn)
                return torch.mm(x_fp8.to(torch.bfloat16), w)

        model = SimpleQuantModule().cuda().to(torch.bfloat16)
        x = torch.randn(16, 10, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(10, 16, device="cuda", dtype=torch.bfloat16)

        expected = model(x, w)
        compiled_model = torch.compile(model)
        actual = compiled_model(x, w)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    run_tests()
