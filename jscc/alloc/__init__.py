"""Per-image budget allocation for the prefix codes (the simpler policy).

    core.py       equal-slope allocation at a matched average rate, statistics
    sweep.py      quality-vs-budget curves of a frozen codec (common random numbers)
    predictor.py  the policy: predict each image's curve, allocate on it
    io.py         curve files

Tools: tools/alloc_sweep.py (curves), tools/alloc_policy.py (oracle, train, eval).
"""
