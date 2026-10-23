from __future__ import annotations

import unittest

from gpu_preflight.compatibility import (
    best_cuda_code_path,
    classify_cuda_code_path,
    evaluate_cuda_target,
)


class CompatibilityTests(unittest.TestCase):
    def test_sm103_cubin_is_binary_compatible_with_sm107(self) -> None:
        result = evaluate_cuda_target("sm_103", 107)
        self.assertTrue(result.compatible)
        self.assertEqual(result.classification, "binary-compatible")

    def test_compute103f_is_family_compatible_with_sm107(self) -> None:
        result = evaluate_cuda_target("compute_103f", 107)
        self.assertTrue(result.compatible)
        self.assertEqual(result.classification, "family-compatible")

    def test_architecture_specific_target_is_exact_only(self) -> None:
        result = evaluate_cuda_target("sm_103a", 107)
        self.assertFalse(result.compatible)
        self.assertEqual(result.classification, "incompatible")

    def test_sm107_is_not_backward_compatible_with_sm103(self) -> None:
        result = evaluate_cuda_target("sm_107", 103)
        self.assertFalse(result.compatible)

    def test_older_ptx_is_forward_compatible(self) -> None:
        result = evaluate_cuda_target("compute_90", 103)
        self.assertTrue(result.compatible)
        self.assertEqual(result.classification, "ptx-jit-compatible")

    def test_code_path_distinguishes_native_optimization_levels(self) -> None:
        self.assertEqual(classify_cuda_code_path("sm_103", 103), "exact-base-cubin")
        self.assertEqual(classify_cuda_code_path("sm_103f", 103), "exact-family-cubin")
        self.assertEqual(classify_cuda_code_path("sm_103a", 103), "exact-architecture-cubin")

    def test_best_code_path_prefers_target_specific_cubin(self) -> None:
        self.assertEqual(
            best_cuda_code_path(("compute_90", "sm_100", "sm_103f"), 103),
            "exact-family-cubin",
        )

    def test_code_path_separates_compatible_cubin_and_ptx_fallback(self) -> None:
        self.assertEqual(classify_cuda_code_path("sm_100", 103), "compatible-base-cubin")
        self.assertEqual(
            classify_cuda_code_path("sm_100f", 103),
            "compatible-family-cubin",
        )
        self.assertEqual(classify_cuda_code_path("compute_90", 103), "ptx-jit")


if __name__ == "__main__":
    unittest.main()
