import unittest

from utils.flowvn_run_naming import next_numeric_run_id


class FlowVNRunNamingTests(unittest.TestCase):
    def test_next_numeric_run_id_ignores_non_numeric_directories(self):
        paths = [
            "results/lightning_logs/version_1",
            "results/lightning_logs/validation",
            "results/lightning_logs/version_12",
            "results/lightning_logs/manual_run",
        ]

        self.assertEqual(next_numeric_run_id(paths), "13")

    def test_next_numeric_run_id_starts_at_one(self):
        self.assertEqual(next_numeric_run_id([]), "1")


if __name__ == "__main__":
    unittest.main()
