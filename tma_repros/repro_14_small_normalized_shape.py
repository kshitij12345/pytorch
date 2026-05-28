"""
Reproducer 14: TMA descriptor store in persistent reduction with small R0_BLOCK

Error: Descriptor block shape must have at least 16 bytes in the last
dimension, but got 2 * 4 = 8 bytes

Root cause: The TMACompatibilityChecker code that validates persistent
reduction block sizes only applies to LOADS (`not self.for_store`). For
STORES in persistent reductions where the output shape is very small
(e.g. layer_norm with normalized_shape=[2]), the store uses R0_BLOCK as
the innermost block_shape dimension, but the minimum-16-bytes check is
skipped for stores. Triton then rejects the descriptor at compile time.

This is the code path (triton.py line ~2898):
    if (self.kernel.persistent_reduction
        and not self.for_store  # <-- BUG: excludes stores!
        and innermost_block_symt in TritonSymbols.reduction_types):

Original test: test/inductor/test_torchinductor_opinfo_properties.py::TestOpInfoPropertiesCUDA::test_determinism_nn_functional_layer_norm_backend_inductor_numerics_cuda_float32

Run: python agent_space/tma_repros/repro_14_small_normalized_shape.py
"""

import torch
import torch._inductor.config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTMASmallReductionStore(TestCase):
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
            "triton.persistent_reductions": True,
        }
    )
    def test_persistent_reduction_store_small_output(self):
        """
        With XBLOCK > numel and a persistent reduction, the output is stored
        via a TMA descriptor with block_shape that uses R0_BLOCK. When output
        shape is very small (e.g., [2] for layer_norm), R0_BLOCK=2, giving
        2*4=8 < 16 bytes.

        The store-side check is missing in the persistent reduction path.
        This might not reproduce locally if the heuristics choose different
        block sizes. See the CI failure for the exact error.
        """
        # Use layer_norm with a tiny normalized_shape to force small R0_BLOCK
        layer_norm = torch.nn.LayerNorm([2]).cuda()

        # Small batch to force persistent reduction
        x = torch.randn(1, 2, device="cuda")
        expected = layer_norm(x)
        compiled_ln = torch.compile(layer_norm)
        actual = compiled_ln(x)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    run_tests()
