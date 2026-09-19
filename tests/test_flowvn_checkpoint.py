import unittest
from pathlib import Path

from utils.flowvn_checkpoint import remap_state_dict_for_expected_keys


class FlowVNCheckpointTests(unittest.TestCase):
    def test_efficiency_benchmark_uses_adaptive_compiled_key_remapping(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "benchmark_flowvn.py").read_text()

        self.assertIn(
            "from utils.flowvn_checkpoint import remap_state_dict_for_expected_keys",
            source,
        )
        self.assertIn(
            "stripped_state = remap_state_dict_for_expected_keys(",
            source,
        )
        self.assertIn("set(model.state_dict())", source)

    def test_compiled_checkpoint_keys_load_into_uncompiled_model(self):
        state = {
            "network.cell.0.activation._orig_mod.interpolator.yk": "activation",
            "network.cell.0.kernel": "kernel",
        }
        expected = {
            "network.cell.0.activation.interpolator.yk",
            "network.cell.0.kernel",
        }

        remapped = remap_state_dict_for_expected_keys(state, expected)

        self.assertEqual(
            remapped,
            {
                "network.cell.0.activation.interpolator.yk": "activation",
                "network.cell.0.kernel": "kernel",
            },
        )

    def test_uncompiled_checkpoint_keys_load_into_compiled_model(self):
        state = {"network.cell.0.activation.interpolator.yk": "activation"}
        expected = {
            "network.cell.0.activation._orig_mod.interpolator.yk",
        }

        remapped = remap_state_dict_for_expected_keys(state, expected)

        self.assertEqual(
            remapped,
            {
                "network.cell.0.activation._orig_mod.interpolator.yk": "activation",
            },
        )

    def test_ambiguous_canonical_expected_keys_are_rejected(self):
        state = {"layer.weight": "weight"}
        expected = {"layer.weight", "layer._orig_mod.weight"}

        with self.assertRaisesRegex(ValueError, "Ambiguous expected state keys"):
            remap_state_dict_for_expected_keys(state, expected)


if __name__ == "__main__":
    unittest.main()
