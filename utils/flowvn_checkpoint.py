from collections import OrderedDict
from collections.abc import Iterable, Mapping


def _canonical_state_key(key: str) -> str:
    return ".".join(part for part in str(key).split(".") if part != "_orig_mod")


def remap_state_dict_for_expected_keys(
    state_dict: Mapping, expected_keys: Iterable[str]
):
    """Map torch.compile wrapper keys to the exact keys expected by a model."""
    expected_by_canonical = {}
    for expected_key in expected_keys:
        canonical = _canonical_state_key(expected_key)
        previous = expected_by_canonical.get(canonical)
        if previous is not None and previous != expected_key:
            raise ValueError(
                "Ambiguous expected state keys share canonical key "
                f"{canonical!r}: {previous!r}, {expected_key!r}"
            )
        expected_by_canonical[canonical] = str(expected_key)

    remapped = OrderedDict()
    for incoming_key, value in state_dict.items():
        canonical = _canonical_state_key(incoming_key)
        target_key = expected_by_canonical.get(canonical, str(incoming_key))
        if target_key in remapped:
            raise ValueError(
                f"Checkpoint keys collide after torch.compile normalization: {target_key!r}"
            )
        remapped[target_key] = value

    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        remapped._metadata = metadata
    return remapped
