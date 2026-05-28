"""
Reproducer 13: AOTI eager mode fails to produce kernel library with TMA

Error: RuntimeError: Failed to produce kernel library by using AOTI for CUDA.
Operator Name is aten::clamp, Overload Name is Tensor_out

Root cause: When TMA is globally enabled, the AOTI eager compilation path
(which compiles individual ops on-the-fly via _impl_with_aoti_compile)
fails because the generated C++ code references `cached_torch_device_type_cuda`
which is undeclared (same root cause as repro_04). This manifests as a complete
failure to produce a .so library for the op.

Original test: test/inductor/test_torchinductor.py::GPUTests::test_aoti_eager_support_out_cuda

Run: python agent_space/tma_repros/repro_13_aoti_eager_failure.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.library import _scoped_library
from torch.testing._internal.common_utils import run_tests, TestCase


def register_ops_with_aoti_compile(ns, op_set, dispatch_key, lib_impl):
    for op_name in op_set:
        qualified_op_name = f"{ns}::{op_name}"
        _, overload_names = torch._C._jit_get_operation(qualified_op_name)
        for overload_name in overload_names:
            try:
                reg_op_name = qualified_op_name
                schema = torch._C._get_schema(qualified_op_name, overload_name)
                if schema.overload_name:
                    reg_op_name = f"{qualified_op_name}.{schema.overload_name}"
                lib_impl._impl_with_aoti_compile(reg_op_name, dispatch_key)
            except Exception:
                continue


class TestTMAAOTIEagerFailure(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_aoti_eager_clamp_out(self):
        device = "cuda"
        inp_tensor = torch.randn(128, dtype=torch.float, device=device).fill_(1.0)
        min_tensor = inp_tensor - 0.05
        max_tensor = inp_tensor + 0.05
        out_tensor = torch.randn(128, dtype=torch.float, device=device).fill_(-1)

        with _scoped_library("aten", "IMPL") as torch_compile_op_lib_impl:
            register_ops_with_aoti_compile(
                "aten", ["clamp"], "CUDA", torch_compile_op_lib_impl
            )
            result = torch.clamp(
                input=inp_tensor, min=min_tensor, max=max_tensor, out=out_tensor
            )
            self.assertTrue(torch.allclose(result, inp_tensor))


if __name__ == "__main__":
    run_tests()
