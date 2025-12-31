# metric_utils.py
import numpy as np
import pandas as pd


def calc_wrs(
    df: pd.DataFrame,
    protected_col: str = None,
    output_col: str = None,
    min_per_group: int = 2
) -> float:
    """
    Compute a simple Weighted Rejection Score (WRS) for a single protected attribute.

    Formula (for a binary protected attribute Z):
        WRS = p * | mean(pred | Z=1) - mean(pred | Z=0) |
    where p = P(Z = 1) (prevalence of group 1).

    Notes:
    - This implementation expects `protected_col` to be provided (no arbitrary default).
    - If the column isn't strictly binary, we coerce values to binary via (val > 0).
      This makes the function robust for many synthetic datasets.
    - If either group has fewer than `min_per_group` samples, returns 0.0.
    - The default output column is "__MODEL_OUTPUT__" to match RatingEnv's normalized column.
    """

    if protected_col is None:
        raise ValueError("protected_col must be provided (e.g. 'Z1').")

    if output_col not in df.columns:
        # Missing output column -> cannot compute
        return 0.0
    
    

    # Extract columns and drop rows where either is missing
    sub = df[[protected_col, output_col]].dropna()
    if len(sub) == 0:
        return 0.0

    z = sub[protected_col]
    y = sub[output_col].astype(float)

    # Coerce protected attribute to binary (treat any positive value as 1)
    unique_vals = set(pd.unique(z))
    if not unique_vals.issubset({0, 1}):
        z = (z > 0).astype(int)
    else:
        z = z.astype(int)

    # If only one group present, no gap
    groups_present = set(z.unique())
    if len(groups_present) < 2:
        return 0.0

    # Ensure minimum per-group count
    n1 = int((z == 1).sum())
    n0 = int((z == 0).sum())
    if n1 < min_per_group or n0 < min_per_group:
        return 0.0

    mean1 = float(y[z == 1].mean())
    mean0 = float(y[z == 0].mean())
    p = float(n1 / (n0 + n1))

    gap = abs(mean1 - mean0)
    wrs = p * gap

    return float(wrs)
