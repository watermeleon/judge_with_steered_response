"""
Correctness statistics and paired significance tests on the human annotations.

Reproduces the response-quality numbers in Section 5 of the paper: mean correctness for
base / provoked / steered-provoked responses and the Wilcoxon signed-rank tests between
them.

Aggregation matters and is reported both ways:
- per question: average the 3 annotators first, then summarise over questions. This is
  what the paper reports, and it is what the paired tests use (the 3 annotators of a
  question are not independent observations, so the question is the unit).
- per annotation: every individual rating as its own observation. Same means, larger
  standard deviations, because annotator noise is no longer averaged out.

Wilcoxon is implemented here rather than imported: scipy is not in requirements.txt.
`wilcoxon_signed_rank` follows scipy.stats.wilcoxon's defaults (zero_method='wilcox',
normal approximation with tie correction, no continuity correction) and is cross-checked
by `sign_flip_p`, a permutation test on the same statistic that assumes nothing.

Usage:
    python -m steering_vec_functions.human_annotation.response_quality_stats
    python -m steering_vec_functions.human_annotation.response_quality_stats --alternative greater
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from steering_vec_functions.human_annotation.annotator_agreement import load_annotations


DEFAULT_LEAVE_OUT = ["false_dichotomy", "false_transparency", "false_causality", "risk_distortion"]

CORRECTNESS_COLUMNS = {
    "base": "correctness__base",
    "provoked": "correctness__suggestive",
    "steered-provoked": "correctness__suggestive_steered",
}

COMPARISONS = [("steered-provoked", "provoked"), ("base", "provoked"), ("steered-provoked", "base")]


def rankdata(values):
    """Ranks with ties averaged (equivalent to scipy.stats.rankdata's 'average')."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def wilcoxon_signed_rank(x, y, alternative="two-sided"):
    """
    Wilcoxon signed-rank test for paired samples, normal approximation.

    Returns dict with the statistic W (the smaller signed-rank sum), z, p, and the number
    of non-zero differences the test is actually based on.
    """
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[diff != 0]  # zero_method='wilcox': discard zero differences
    n = len(diff)
    if n == 0:
        return {"W": float("nan"), "z": float("nan"), "p": float("nan"), "n_nonzero": 0}

    ranks = rankdata(np.abs(diff))
    w_plus = ranks[diff > 0].sum()
    w_minus = ranks[diff < 0].sum()
    statistic = min(w_plus, w_minus)

    mean = n * (n + 1) / 4.0
    _, tie_counts = np.unique(np.abs(diff), return_counts=True)
    tie_correction = (tie_counts ** 3 - tie_counts).sum()
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 48.0)
    z = (statistic - mean) / sd

    two_sided = math.erfc(abs(z) / math.sqrt(2))
    if alternative == "two-sided":
        p = two_sided
    elif alternative in ("greater", "less"):
        # one-sided in the direction of the observed effect, halved as usual
        favours = (w_plus > w_minus) if alternative == "greater" else (w_minus > w_plus)
        p = two_sided / 2.0 if favours else 1.0 - two_sided / 2.0
    else:
        raise ValueError(f"unknown alternative: {alternative!r}")

    return {"W": float(statistic), "z": float(z), "p": float(p), "n_nonzero": int(n)}


def sign_flip_p(x, y, n_perm=200000, seed=0):
    """
    Permutation p-value for the same signed-rank statistic: flip the sign of each
    difference at random. Distribution-free, so it validates the normal approximation.
    Two-sided.
    """
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return float("nan")
    ranks = rankdata(np.abs(diff))
    observed = ranks[diff > 0].sum()
    centre = ranks.sum() / 2.0

    rng = np.random.default_rng(seed)
    null = (rng.random((n_perm, len(diff))) < 0.5) @ ranks
    return float((np.abs(null - centre) >= abs(observed - centre)).mean())


def per_question_correctness(df):
    """One row per question, correctness averaged over that question's annotators."""
    return df.groupby("question_id")[list(CORRECTNESS_COLUMNS.values())].mean()


def correctness_summary(df):
    """Mean / std of correctness per model, aggregated per question and per annotation."""
    per_question = per_question_correctness(df)
    rows = []
    for name, column in CORRECTNESS_COLUMNS.items():
        rows.append({
            "model": name,
            "per_question_mean": per_question[column].mean(),
            "per_question_std": per_question[column].std(ddof=1),
            "per_annotation_mean": df[column].mean(),
            "per_annotation_std": df[column].std(ddof=1),
            "n_questions": int(per_question[column].notna().sum()),
            "n_ratings": int(df[column].notna().sum()),
        })
    return pd.DataFrame(rows)


def paired_tests(df, alternative="two-sided", n_perm=200000, seed=0):
    """Wilcoxon signed-rank between every model pair, on per-question mean correctness."""
    per_question = per_question_correctness(df)
    rows = []
    for a, b in COMPARISONS:
        xa, xb = per_question[CORRECTNESS_COLUMNS[a]], per_question[CORRECTNESS_COLUMNS[b]]
        result = wilcoxon_signed_rank(xa, xb, alternative=alternative)
        rows.append({
            "comparison": f"{a} vs {b}",
            "mean_difference": xa.mean() - xb.mean(),
            **result,
            "permutation_p_two_sided": sign_flip_p(xa, xb, n_perm=n_perm, seed=seed),
        })
    return pd.DataFrame(rows)


def report(df, scope_name, alternative="two-sided", n_perm=200000, seed=0):
    print("\n" + "=" * 88)
    print(f"{scope_name}  ({df['question_id'].nunique()} questions, {len(df)} annotations)")
    print("=" * 88)

    summary = correctness_summary(df)
    print("correctness (0-5):")
    print(f"  {'model':<20}{'per-question mean±std':>26}{'per-annotation mean±std':>28}")
    for _, r in summary.iterrows():
        print(f"  {r['model']:<20}"
              f"{r['per_question_mean']:>17.3f} ± {r['per_question_std']:<6.3f}"
              f"{r['per_annotation_mean']:>19.3f} ± {r['per_annotation_std']:<6.3f}")

    tests = paired_tests(df, alternative=alternative, n_perm=n_perm, seed=seed)
    print(f"\nWilcoxon signed-rank on per-question means (alternative={alternative}):")
    print(f"  {'comparison':<34}{'Δmean':>8}{'W':>10}{'z':>8}{'p':>12}{'perm p':>11}{'n':>5}")
    for _, r in tests.iterrows():
        print(f"  {r['comparison']:<34}{r['mean_difference']:>+8.3f}{r['W']:>10.1f}"
              f"{r['z']:>8.3f}{r['p']:>12.2e}{r['permutation_p_two_sided']:>11.1e}"
              f"{r['n_nonzero']:>5}")
    return summary, tests


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leave_out_cats", nargs="*", default=DEFAULT_LEAVE_OUT,
                        help="categories excluded in the 'kept' scope")
    parser.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater", "less"])
    parser.add_argument("--n_perm", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_annotations(verbose=True)
    report(df, "ALL 13 CATEGORIES", alternative=args.alternative, n_perm=args.n_perm, seed=args.seed)
    if args.leave_out_cats:
        kept = df[~df["category"].isin(args.leave_out_cats)]
        report(kept, f"KEPT CATEGORIES (excluding {sorted(args.leave_out_cats)})",
               alternative=args.alternative, n_perm=args.n_perm, seed=args.seed)


if __name__ == "__main__":
    main()
