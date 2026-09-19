import ast
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_function(fake_torch):
    tree = ast.parse((ROOT / "main.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_set_torchdynamo_lru_cache"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": fake_torch}
    exec(compile(module, "main.py", "exec"), namespace)
    return namespace["_set_torchdynamo_lru_cache"]


class TorchDynamoLruCompatibilityTests(unittest.TestCase):
    def test_uses_private_setter_when_available(self):
        calls = []
        eval_frame = types.SimpleNamespace(
            _set_lru_cache=lambda enabled: calls.append(enabled)
        )
        fake_torch = types.SimpleNamespace(
            _C=types.SimpleNamespace(
                _dynamo=types.SimpleNamespace(eval_frame=eval_frame)
            ),
            _dynamo=types.SimpleNamespace(
                config=types.SimpleNamespace(
                    cache_size_limit=8,
                    accumulated_cache_size_limit=256,
                )
            ),
        )

        mode = _load_function(fake_torch)(False)

        self.assertEqual(calls, [False])
        self.assertEqual(mode, "private_lru")

    def test_falls_back_to_supported_cache_limits_when_private_setter_is_absent(self):
        config = types.SimpleNamespace(
            cache_size_limit=8,
            accumulated_cache_size_limit=256,
        )
        fake_torch = types.SimpleNamespace(
            _C=types.SimpleNamespace(
                _dynamo=types.SimpleNamespace(eval_frame=types.SimpleNamespace())
            ),
            _dynamo=types.SimpleNamespace(config=config),
        )

        mode = _load_function(fake_torch)(False)

        self.assertEqual(mode, "expanded_limits")
        self.assertGreaterEqual(config.cache_size_limit, 64)
        self.assertGreaterEqual(config.accumulated_cache_size_limit, 1024)


if __name__ == "__main__":
    unittest.main()
