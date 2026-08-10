"""
Inter-annotator agreement metrics.

Self-contained implementations (numpy only) of:
- Krippendorff's alpha (nominal / ordinal / interval / ratio difference functions)
- Fleiss' kappa
- observed (percent) agreement over rater pairs

Krippendorff's alpha is used as the main statistic because the annotation design is
incomplete: every question is rated by 3 annotators, but each annotator only rates a
subset of the questions, so raters are not exchangeable columns of a full matrix.
Alpha is defined on the coincidence matrix and therefore handles missing cells and a
varying number of ratings per unit without dropping data.

Reliability data convention used throughout this module:
    a 2D array of shape (n_raters, n_units), with np.nan for "this rater did not rate
    this unit". Values must be numeric; map categorical labels to integer codes first
    (see `codes_from_labels`).
"""

import numpy as np


# --------------------------------------------------------------------------------------
# difference functions delta^2(c, k)
# --------------------------------------------------------------------------------------

def _delta_nominal(values, counts):
    v = np.asarray(values, dtype=float)
    return (v[:, None] != v[None, :]).astype(float)


def _delta_interval(values, counts):
    v = np.asarray(values, dtype=float)
    return (v[:, None] - v[None, :]) ** 2


def _delta_ratio(values, counts):
    v = np.asarray(values, dtype=float)
    denom = v[:, None] + v[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(denom == 0, 0.0, ((v[:, None] - v[None, :]) / denom) ** 2)
    return d


def _delta_ordinal(values, counts):
    """
    Ordinal difference: delta^2(c, k) = ( sum_{g=c..k} n_g - (n_c + n_k) / 2 )^2,
    where n_g are the marginal frequencies of the coincidence matrix and the values are
    taken in increasing rank order.
    """
    n = np.asarray(counts, dtype=float)
    cum = np.cumsum(n)
    # sum_{g=c..k} n_g for c <= k, using cumulative sums
    inclusive = cum[None, :] - cum[:, None] + n[:, None]
    inclusive = np.where(inclusive < 0, 0.0, inclusive)  # zeroed for c > k, filled below
    d = (inclusive - (n[:, None] + n[None, :]) / 2.0) ** 2
    d = np.triu(d) + np.triu(d, 1).T  # symmetrise from the upper triangle (c <= k)
    np.fill_diagonal(d, 0.0)
    return d


_DELTAS = {
    "nominal": _delta_nominal,
    "ordinal": _delta_ordinal,
    "interval": _delta_interval,
    "ratio": _delta_ratio,
}


# --------------------------------------------------------------------------------------
# Krippendorff's alpha
# --------------------------------------------------------------------------------------

def coincidence_matrix(reliability_data):
    """
    Build the coincidence matrix from reliability data.

    Returns (values, matrix) where `values` are the sorted distinct observed values and
    `matrix[c, k]` counts the value pairs (c, k) within units, each pair weighted by
    1 / (m_u - 1) with m_u the number of ratings of unit u. Units rated only once carry
    no pairable information and are skipped.
    """
    data = np.asarray(reliability_data, dtype=float)
    if data.ndim != 2:
        raise ValueError("reliability_data must be 2D (n_raters, n_units)")

    values = np.unique(data[~np.isnan(data)])
    index = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)), dtype=float)

    for unit in data.T:
        unit = unit[~np.isnan(unit)]
        m_u = len(unit)
        if m_u < 2:
            continue
        # with c_v ratings of value v in this unit, the ordered pairs (i != j) taking
        # values (a, b) number c_a * c_b, minus the c_a self-pairs when a == b
        counts = np.bincount([index[v] for v in unit], minlength=len(values)).astype(float)
        o += (np.outer(counts, counts) - np.diag(counts)) / (m_u - 1)
    return values, o


def krippendorff_alpha(reliability_data, level="nominal", value_domain=None):
    """
    Krippendorff's alpha for the given reliability data.

    Args:
        reliability_data: (n_raters, n_units) array, np.nan where a rater did not rate.
        level: 'nominal', 'ordinal', 'interval' or 'ratio'.
        value_domain: optional sorted sequence of all admissible values. Only affects the
            ordinal metric, and only through values that never occur (they get count 0),
            so it is normally safe to leave as None.

    Returns:
        float alpha, or np.nan when fewer than two pairable values are available.
        alpha = 1 is perfect agreement, 0 is chance-level, negative is systematic
        disagreement.
    """
    if level not in _DELTAS:
        raise ValueError(f"level must be one of {sorted(_DELTAS)}, got {level!r}")

    values, o = coincidence_matrix(reliability_data)
    if len(values) == 0:
        return float("nan")

    if value_domain is not None:
        domain = np.asarray(sorted(set(np.asarray(value_domain, dtype=float)) | set(values)))
        full = np.zeros((len(domain), len(domain)))
        pos = {v: i for i, v in enumerate(domain)}
        for i, vi in enumerate(values):
            for j, vj in enumerate(values):
                full[pos[vi], pos[vj]] = o[i, j]
        values, o = domain, full

    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n < 2:
        return float("nan")
    if len(values) == 1:
        # every rater gave the same value everywhere: no variation, agreement is perfect
        return 1.0

    delta = _DELTAS[level](values, n_c)
    observed = (o * delta).sum() / n
    expected = (np.outer(n_c, n_c) - np.diag(n_c)) * delta
    expected = expected.sum() / (n * (n - 1))

    if expected == 0:
        return float("nan")
    return 1.0 - observed / expected


def bootstrap_alpha_ci(reliability_data, level="nominal", n_boot=2000, alpha_level=0.05,
                       seed=0):
    """
    Percentile confidence interval for Krippendorff's alpha by resampling *units*
    (columns) with replacement. Units are the independent observations here, so this is
    the appropriate resampling unit; it keeps each unit's set of ratings intact.

    Returns (low, high), or (nan, nan) if too few units resolve.
    """
    data = np.asarray(reliability_data, dtype=float)
    n_units = data.shape[1]
    if n_units < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        cols = rng.integers(0, n_units, size=n_units)
        a = krippendorff_alpha(data[:, cols], level=level)
        if not np.isnan(a):
            draws.append(a)
    if len(draws) < 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(draws, [100 * alpha_level / 2, 100 * (1 - alpha_level / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------------------
# Fleiss' kappa and percent agreement
# --------------------------------------------------------------------------------------

def fleiss_kappa(reliability_data):
    """
    Fleiss' kappa on nominal reliability data.

    Fleiss' kappa assumes the same number of ratings per unit; units whose rating count
    differs from the modal count are dropped (a count is reported by the caller through
    `pairwise_percent_agreement`/`n_units` if needed). Raters are treated as
    interchangeable, which fits this design where the 3 annotators of a question are an
    arbitrary draw from the annotator pool.
    """
    data = np.asarray(reliability_data, dtype=float)
    counts_per_unit = (~np.isnan(data)).sum(axis=0)
    if counts_per_unit.max(initial=0) < 2:
        return float("nan")
    n_raters = int(np.bincount(counts_per_unit).argmax())
    if n_raters < 2:
        return float("nan")

    keep = counts_per_unit == n_raters
    data = data[:, keep]
    values = np.unique(data[~np.isnan(data)])
    if len(values) < 2:
        return 1.0 if len(values) == 1 else float("nan")

    table = np.zeros((data.shape[1], len(values)))
    for j, v in enumerate(values):
        table[:, j] = (data == v).sum(axis=0)

    n_units = table.shape[0]
    p_j = table.sum(axis=0) / (n_units * n_raters)
    p_i = ((table ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_e = (p_j ** 2).sum()
    if p_e == 1:
        return float("nan")
    return float((p_bar - p_e) / (1 - p_e))


def pairwise_percent_agreement(reliability_data):
    """
    Fraction of within-unit rater pairs that gave the identical value.
    Not chance-corrected; useful as a raw sanity check next to alpha.
    """
    data = np.asarray(reliability_data, dtype=float)
    agree = 0.0
    total = 0.0
    for unit in data.T:
        unit = unit[~np.isnan(unit)]
        if len(unit) < 2:
            continue
        for i in range(len(unit)):
            for j in range(i + 1, len(unit)):
                total += 1
                agree += float(unit[i] == unit[j])
    if total == 0:
        return float("nan")
    return agree / total


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def codes_from_labels(labels):
    """
    Map a sequence of hashable labels (None/NaN allowed) to a sorted integer coding.
    Returns (codes array of float with nan for missing, list of labels in code order).
    """
    def _missing(x):
        return x is None or (isinstance(x, float) and np.isnan(x))

    uniques = sorted({x for x in labels if not _missing(x)}, key=str)
    lookup = {lab: float(i) for i, lab in enumerate(uniques)}
    codes = np.array([np.nan if _missing(x) else lookup[x] for x in labels], dtype=float)
    return codes, uniques


def build_reliability_matrix(rater_ids, unit_ids, values):
    """
    Build a (n_raters, n_units) reliability matrix from long-format columns.

    Duplicate (rater, unit) pairs are not expected; if one occurs the last value wins.
    Returns (matrix, raters, units) with `raters`/`units` giving the row/column labels.
    """
    rater_ids = list(rater_ids)
    unit_ids = list(unit_ids)
    values = list(values)
    if not (len(rater_ids) == len(unit_ids) == len(values)):
        raise ValueError("rater_ids, unit_ids and values must have the same length")

    raters = sorted(set(rater_ids), key=str)
    units = sorted(set(unit_ids), key=str)
    r_pos = {r: i for i, r in enumerate(raters)}
    u_pos = {u: i for i, u in enumerate(units)}

    matrix = np.full((len(raters), len(units)), np.nan)
    for r, u, v in zip(rater_ids, unit_ids, values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        matrix[r_pos[r], u_pos[u]] = float(v)
    return matrix, raters, units


def agreement_report(rater_ids, unit_ids, values, level="nominal", n_boot=0, seed=0):
    """
    Convenience wrapper: build the matrix from long format and return every statistic.

    Returns a dict with alpha, its bootstrap CI (when n_boot > 0), Fleiss' kappa,
    percent agreement and the counts the numbers rest on.
    """
    matrix, raters, units = build_reliability_matrix(rater_ids, unit_ids, values)
    n_per_unit = (~np.isnan(matrix)).sum(axis=0)
    pairable = int((n_per_unit >= 2).sum())

    out = {
        "alpha": float(krippendorff_alpha(matrix, level=level)),
        "level": level,
        "fleiss_kappa": float(fleiss_kappa(matrix)) if level == "nominal" else None,
        "percent_agreement": float(pairwise_percent_agreement(matrix)),
        "n_units": int(matrix.shape[1]),
        "n_pairable_units": pairable,
        "n_raters": int(len(raters)),
        "n_ratings": int((~np.isnan(matrix)).sum()),
    }
    if n_boot:
        lo, hi = bootstrap_alpha_ci(matrix, level=level, n_boot=n_boot, seed=seed)
        out["alpha_ci_low"], out["alpha_ci_high"] = lo, hi
    return out


# --------------------------------------------------------------------------------------
# self-test against the published worked example
# --------------------------------------------------------------------------------------

def _self_test():
    """Reference example from Krippendorff (2011), 'Computing Krippendorff's Alpha'."""
    nan = np.nan
    data = np.array([
        [nan, nan, nan, nan, nan, 3, 4, 1, 2, 1, 1, 3, 3, nan, 3],
        [1, nan, 2, 1, 3, 3, 4, 3, nan, nan, nan, nan, nan, nan, nan],
        [nan, nan, 2, 1, 3, 4, 4, nan, 2, 1, 1, 3, 3, nan, 4],
    ], dtype=float)

    expected = {"nominal": 0.691, "ordinal": 0.807, "interval": 0.811}
    for level, target in expected.items():
        got = krippendorff_alpha(data, level=level)
        assert abs(got - target) < 5e-4, f"{level}: got {got:.4f}, expected {target}"
        print(f"  alpha ({level:8s}) = {got:.4f}  (reference {target})")

    # Fleiss' kappa on a complete 3-rater / 2-category table
    complete = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=float)
    k = fleiss_kappa(complete)
    assert -1 <= k <= 1
    assert abs(pairwise_percent_agreement(complete) - 8 / 12) < 1e-12

    # perfect agreement and perfect disagreement bounds
    assert abs(krippendorff_alpha(np.array([[1.0, 2, 3], [1, 2, 3]])) - 1.0) < 1e-12
    assert krippendorff_alpha(np.array([[1.0, 1, 1], [2, 2, 2]])) < 0
    print("  self-test passed")


if __name__ == "__main__":
    _self_test()
