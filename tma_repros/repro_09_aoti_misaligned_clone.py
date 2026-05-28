"""
Reproducer 9: AOTI misaligned input generates unwanted clone

Error: RuntimeError: Expected to not find "aoti_torch_clone_preserve_strides"
but found it

Root cause: When TMA is enabled, the AOTInductor wrapper generates alignment
checking code that copies misaligned inputs to aligned buffers. The test
`test_misaligned_input_2` specifically checks that a misaligned input with
offset NOT divisible by element size should NOT produce a clone. With TMA,
the 16-byte alignment requirement is stricter, causing the wrapper to insert
copies for inputs that were previously considered acceptable.

NOTE: This reproducer may pass on some hardware/configs where the model is
simple enough that TMA is not used for the kernel. The original CI failure
was on a more complex model. The key behavioral difference is that with TMA
enabled, `assume_aligned_inputs=True` causes the wrapper to add runtime
alignment checks + clone fallbacks.

Original test: test/inductor/test_aot_inductor.py::AOTInductorTestABICompatibleGpu::test_misaligned_input_2_cuda

Run: python agent_space/tma_repros/repro_09_aoti_misaligned_clone.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMAAOTIMisalignedClone(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_aoti_misaligned_input_generates_clone(self):
        """
        With TMA + assume_aligned_inputs, AOTI generates code that checks
        alignment at runtime and clones misaligned inputs. This test verifies
        this behavior exists (the original CI test asserts it should NOT).
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.sin() + x.cos()

        model = Model().cuda()
        base = torch.randn(129, device="cuda", dtype=torch.float16)
        x = base[1:]  # 2 bytes offset, not 16-byte aligned

        import tempfile
        import os
        import glob

        with tempfile.TemporaryDirectory() as tmpdir:
            so_path = torch._export.aot_compile(
                model,
                (x,),
                options={"aot_inductor.output_path": f"{tmpdir}/model.so"},
            )
            # Check the generated wrapper cpp for clone code
            cpp_files = glob.glob(f"{tmpdir}/**/*.cpp", recursive=True)
            found_clone = False
            for cpp_file in cpp_files:
                with open(cpp_file) as f:
                    if "clone_preserve_strides" in f.read():
                        found_clone = True
                        break

            # TMA forces alignment checks which generate clone fallbacks
            # The original test asserts this should NOT happen
            self.assertFalse(
                found_clone,
                "TMA + assume_aligned_inputs should not generate "
                "clone_preserve_strides for misaligned inputs "
                "(but currently does, breaking the original test)",
            )


if __name__ == "__main__":
    run_tests()
