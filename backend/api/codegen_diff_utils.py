from typing import Dict


def parse_git_numstat_output(raw_output: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for line in (raw_output or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ins = int(parts[0]) if parts[0].isdigit() else 0
        dels = int(parts[1]) if parts[1].isdigit() else 0
        out[parts[2]] = {"insertions": ins, "deletions": dels}
    return out
