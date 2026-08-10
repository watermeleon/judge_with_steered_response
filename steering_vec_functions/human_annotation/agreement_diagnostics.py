"""
Diagnostics that reconcile the low Krippendorff's alpha with the high headline percentages
reported for the human evaluation.

Alpha asks whether annotators agree on *which questions* are manipulative, beyond their
shared tendency. The headline percentages are *marginal* rates. Both can be right at once,
and this script quantifies exactly how:

1. marginal rates    - how often an individual annotation says the provoked response is
                       more manipulative, overall and per annotator.
2. headline rates    - the per-category majority-vote percentages of Figure 3, and their
                       mean over the kept categories (the 92.22% / 88.89% numbers).
3. independence check- observed pairwise agreement vs what independent annotators sharing
                       the same marginal would produce.
4. null simulation   - 3 independent annotators with that marginal and NO item-level
                       signal: what alpha and what majority rate does that alone give?
5. cross-method tests- do humans and the LLM pair judge track the same per-question and
                       per-category variation? Permutation tested, so it needs no scipy.

Usage:
    python -m steering_vec_functions.human_annotation.agreement_diagnostics
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from steering_vec_functions.human_annotation.agreement_metrics import krippendorff_alpha
from steering_vec_functions.human_annotation.annotator_agreement import (
    load_annotations,
    normalize_text,
)


DEFAULT_LEAVE_OUT = ["false_dichotomy", "false_transparency", "false_causality", "risk_distortion"]
DEFAULT_JUDGE_FILE = "results/final_judge_results/judged_responses_manipulation_GPT4Base.json"

# both variables are coded 1 = provoked judged MORE manipulative than the comparison
CLAIMS = {
    "provoked > base": "base_vs_suggestive",
    "provoked > steered-provoked": "steered_vs_suggestive",
}


def spearman(a, b):
    ranks_a, ranks_b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def marginal_rates(df, column, leave_out_cats):
    kept = df[~df["category"].isin(leave_out_cats)]
    return {"per_annotation_all": float(df[column].mean()),
            "per_annotation_kept": float(kept[column].mean()),
            "n_all": int(df[column].notna().sum()),
            "n_kept": int(kept[column].notna().sum())}


def headline_percentages(df, column, leave_out_cats, n_boot=5000, seed=0):
    """
    Figure 3's human column: per category, the % of questions whose annotator majority says
    the provoked response is more manipulative. The headline is the mean over kept
    categories. Bootstrap resamples questions within each category.
    """
    per_question = df.groupby(["category", "question_id"])[column].mean().reset_index()
    per_question["majority"] = (per_question[column] > 0.5).astype(float)
    per_category = per_question.groupby("category")["majority"].mean() * 100

    kept = per_question[~per_question["category"].isin(leave_out_cats)]
    point = kept.groupby("category")["majority"].mean().mean() * 100

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample = kept.groupby("category")["majority"].apply(
            lambda g: g.iloc[rng.integers(0, len(g), len(g))].mean())
        draws.append(sample.mean() * 100)
    return per_category, point, (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def independence_check(df, column):
    """Observed within-question pairwise agreement vs the independent-annotators baseline."""
    p = df[column].mean()
    agree = total = 0
    for _, group in df.groupby("question_id"):
        values = list(group[column].dropna())
        for i, j in itertools.combinations(range(len(values)), 2):
            total += 1
            agree += int(values[i] == values[j])
    return float(p), agree / total, float(p ** 2 + (1 - p) ** 2)


def null_simulation(df, column, n_sim=2000, seed=0):
    """3 independent annotators sharing the observed marginal, with no item-level signal."""
    p = df[column].mean()
    n_units = df["question_id"].nunique()
    rng = np.random.default_rng(seed)
    alphas, majorities = [], []
    for _ in range(n_sim):
        sim = rng.binomial(1, p, size=(3, n_units)).astype(float)
        alphas.append(krippendorff_alpha(sim, level="nominal"))
        majorities.append((sim.mean(axis=0) > 0.5).mean())
    observed_majority = float((df.groupby("question_id")[column].mean() > 0.5).mean())
    observed_alpha = float(krippendorff_alpha(
        df.pivot_table(index="rater", columns="question_id", values=column).values, level="nominal"))
    return {
        "p": float(p),
        "sim_alpha": (float(np.mean(alphas)), float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))),
        "sim_majority": (float(np.mean(majorities)), float(np.percentile(majorities, 2.5)), float(np.percentile(majorities, 97.5))),
        "observed_alpha": observed_alpha,
        "observed_majority": observed_majority,
    }


def cross_method_tests(df, judge_file, n_perm=20000, seed=0):
    """Do humans and the LLM pair judge track the same variation, per category and per question?"""
    with open(judge_file, "r") as f:
        responses = json.load(f)["responses"]
    llm = pd.DataFrame([{
        "qnorm": normalize_text(r["full_data"]["question_text"]),
        "category": r["full_data"]["category_id"],
        "llm_base": r["judge_base_vs_suggestive"]["response_A"]["metric_score"],
        "llm_provoked": r["judge_base_vs_suggestive"]["response_B"]["metric_score"],
    } for r in responses])

    df = df.copy()
    df["qnorm"] = df["question_text"].apply(normalize_text)
    human = df.groupby(["qnorm", "category"])["base_vs_suggestive"].mean().reset_index()
    human["human_majority"] = (human["base_vs_suggestive"] > 0.5).astype(int)

    merged = human.merge(llm, on=["qnorm", "category"], how="left")
    merged["llm_provoked_more"] = np.where(
        merged.llm_provoked > merged.llm_base, 1.0,
        np.where(merged.llm_provoked < merged.llm_base, 0.0, np.nan))

    rng = np.random.default_rng(seed)

    per_category = merged.groupby("category").agg(
        human=("human_majority", "mean"), llm=("llm_provoked_more", "mean")).reset_index()
    rho = spearman(per_category.human, per_category.llm)
    null = [spearman(per_category.human, rng.permutation(per_category.llm.values))
            for _ in range(n_perm // 4)]
    cat_p = float((np.abs(null) >= abs(rho)).mean())

    q = merged.dropna(subset=["llm_provoked_more"])
    observed = float((q.human_majority == q.llm_provoked_more).mean())
    p_h, p_l = q.human_majority.mean(), q.llm_provoked_more.mean()
    chance = float(p_h * p_l + (1 - p_h) * (1 - p_l))
    null_q = [(q.human_majority.values == rng.permutation(q.llm_provoked_more.values)).mean()
              for _ in range(n_perm)]
    q_p = float((np.array(null_q) >= observed).mean())

    return {"n_matched": int(merged.llm_base.notna().sum()), "per_category": per_category,
            "rho": rho, "rho_p": cat_p, "n_questions": len(q), "agreement": observed,
            "chance": chance, "agreement_p": q_p}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leave_out_cats", nargs="*", default=DEFAULT_LEAVE_OUT)
    parser.add_argument("--judge_file", default=DEFAULT_JUDGE_FILE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_annotations(verbose=True)
    leave_out = args.leave_out_cats or []

    for claim, column in CLAIMS.items():
        print("\n" + "=" * 84)
        print(f"CLAIM: {claim}   (1 = provoked judged more manipulative)")
        print("=" * 84)

        rates = marginal_rates(df, column, leave_out)
        print(f"  per-annotation rate, all 13 categories : {rates['per_annotation_all']:.3f} "
              f"(n={rates['n_all']})")
        print(f"  per-annotation rate, kept categories   : {rates['per_annotation_kept']:.3f} "
              f"(n={rates['n_kept']})")

        per_category, point, ci = headline_percentages(df, column, leave_out, seed=args.seed)
        print("\n  per-category % of questions where the annotator majority agrees:")
        for category, value in per_category.items():
            flag = "  (excluded)" if category in leave_out else ""
            print(f"    {category:<26}{value:5.0f}%{flag}")
        print(f"  >>> mean over kept categories: {point:.2f}%  95% CI [{ci[0]:.1f}, {ci[1]:.1f}]")

    print("\n" + "=" * 84)
    print("WHY ALPHA IS ~0 DESPITE THOSE PERCENTAGES")
    print("=" * 84)
    column = CLAIMS["provoked > base"]

    per_rater = df.groupby("rater")[column].mean()
    print(f"  per-annotator rates: mean {per_rater.mean():.3f}, "
          f"min {per_rater.min():.3f}, max {per_rater.max():.3f}")
    print(f"  annotators above chance: {(per_rater > 0.5).sum()}/{len(per_rater)}")

    p, observed, expected = independence_check(df, column)
    print(f"\n  marginal p                             = {p:.3f}")
    print(f"  observed pairwise agreement            = {observed:.3f}")
    print(f"  expected if annotators independent     = {expected:.3f}")
    print("  -> essentially all observed agreement is explained by the shared marginal,")
    print("     which is exactly what alpha ~ 0 reports (the prevalence problem).")

    sim = null_simulation(df, column, seed=args.seed)
    print(f"\n  null simulation: 3 independent annotators at p={sim['p']:.3f}, NO item-level signal")
    print(f"    simulated alpha         = {sim['sim_alpha'][0]:+.3f} "
          f"[{sim['sim_alpha'][1]:+.2f}, {sim['sim_alpha'][2]:+.2f}]")
    print(f"    simulated majority rate = {sim['sim_majority'][0]:.3f} "
          f"[{sim['sim_majority'][1]:.3f}, {sim['sim_majority'][2]:.3f}]")
    print(f"    OBSERVED alpha          = {sim['observed_alpha']:+.3f}")
    print(f"    OBSERVED majority rate  = {sim['observed_majority']:.3f}")

    if os.path.exists(args.judge_file):
        print("\n" + "=" * 84)
        print("BUT ITEM-LEVEL SIGNAL DOES EXIST - IT SHOWS UP ONCE YOU POOL")
        print("=" * 84)
        cm = cross_method_tests(df, args.judge_file, seed=args.seed)
        print(f"  matched {cm['n_matched']} questions to the LLM pair judge\n")
        print("  per-category rate 'provoked more manipulative':")
        print(cm["per_category"].round(3).to_string(index=False))
        print(f"\n  category level: Spearman(human, LLM) = {cm['rho']:.3f}, "
              f"permutation p = {cm['rho_p']:.4f}")
        print(f"  question level: human majority vs LLM agree = {cm['agreement']:.3f} "
              f"(n={cm['n_questions']} non-tied)")
        print(f"                  chance given marginals      = {cm['chance']:.3f}")
        print(f"                  permutation p               = {cm['agreement_p']:.4f}")
    else:
        print(f"\n  (skipping cross-method tests: {args.judge_file} not found)")


if __name__ == "__main__":
    main()
