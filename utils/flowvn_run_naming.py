from pathlib import Path


def next_numeric_run_id(paths) -> str:
    run_ids = []
    for path in paths:
        suffix = Path(path).name.rsplit("_", 1)[-1]
        if suffix.isdecimal():
            run_ids.append(int(suffix))
    return str(max(run_ids, default=0) + 1)
