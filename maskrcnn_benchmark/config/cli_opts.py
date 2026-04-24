# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Helpers for yacs cfg.merge_from_list() when values come from shell argv (all strings)."""


def coerce_yacs_cli_opts(opts):
    """
    Pairwise KEY VALUE list from argparse REMAINDER: coerce lowercase true/false to bool
    so yacs does not raise Type mismatch for bool keys (e.g. DATASETS.*_STRICT true).
    """
    if opts is None or len(opts) == 0:
        return []
    out = []
    n = len(opts)
    i = 0
    while i < n:
        if i + 1 >= n:
            out.append(opts[i])
            break
        k, v = opts[i], opts[i + 1]
        if isinstance(v, str):
            lv = v.lower()
            if lv == "true":
                v = True
            elif lv == "false":
                v = False
        out.extend([k, v])
        i += 2
    return out
