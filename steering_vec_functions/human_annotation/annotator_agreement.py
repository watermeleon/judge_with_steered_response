"""
Inter-annotator agreement for the human annotation study (manipulation dataset).

Each question was shown to 3 annotators together with three responses (base, suggestive,
suggestive_steered, in a per-question fixed but shuffled order). Annotators marked the
most and the least manipulative response and rated the correctness of all three (0-5).
Every annotator saw only a subset of the questions, so the design is incomplete and
Krippendorff's alpha is the statistic of choice (see `agreement_metrics`).

Agreement is reported for six variables:

  question-level units (one unit per question, 3 ratings each)
    most_manipulative      which model was marked most manipulative        (nominal)
    least_manipulative     which model was marked least manipulative       (nominal)
    base_vs_suggestive     is base less manipulative than suggestive?      (nominal/binary)
    steered_vs_suggestive  is suggestive_steered less manipulative
                           than suggestive?                                (nominal/binary)

  response-level units (one unit per question x model, 3 ratings each)
    manipulativeness_rank  3 = most, 2 = middle, 1 = least manipulative    (ordinal)
    correctness            0-5 correctness rating                          (ordinal)

Everything is reported per manipulation category, per higher-level category, per split
and over all 130 questions combined.

Usage:
    python -m steering_vec_functions.human_annotation.annotator_agreement
    python -m steering_vec_functions.human_annotation.annotator_agreement --n_boot 0
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from steering_vec_functions.human_annotation.agreement_metrics import agreement_report


DEFAULT_SPLIT_FILES = {
    "A": "results/human_annotation/manip_split_A_results.xlsx",
    "B": "results/human_annotation/manip_split_B_results.xlsx",
}
DEFAULT_DATASET_JSON = "steering_vec_functions/manipulation_data/manipulation_dataset.json"
DEFAULT_OUTPUT = "results/human_annotation/inter_annotator_agreement.json"

MODELS = ["base", "suggestive", "suggestive_steered"]
POSITIONS = ["a", "b", "c"]

# variable name -> (measurement level, unit level)
VARIABLES = {
    "most_manipulative": ("nominal", "question"),
    "least_manipulative": ("nominal", "question"),
    "base_vs_suggestive": ("nominal", "question"),
    "steered_vs_suggestive": ("nominal", "question"),
    "manipulativeness_rank": ("ordinal", "response"),
    "correctness": ("ordinal", "response"),
}


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def normalize_text(text):
    """Whitespace/case normalisation, so question strings match across files."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def clean_responses(df, metric_name="manipulative", verbose=True):
    """
    Same quality filter as `notebooks/load_human_annoation_results.ipynb`: drop rows with
    missing required fields, and rows where the most and least manipulative response are
    the same. Kept here so the agreement numbers are computed on exactly the rows the
    rest of the analysis uses.
    """
    required = [f"most_{metric_name}_response", f"least_{metric_name}_response"]
    required += [f"correctness_{p}" for p in POSITIONS]

    missing = df[required].isnull().any(axis=1)
    same = (~missing) & (df[f"most_{metric_name}_response"] == df[f"least_{metric_name}_response"])
    clean = df.loc[~(missing | same)].copy()

    if verbose:
        print(f"  rows: {len(df)} -> {len(clean)} "
              f"(dropped {int(missing.sum())} incomplete, {int(same.sum())} most == least)")
    return clean


def load_category_map(dataset_json=DEFAULT_DATASET_JSON, judge_file=None):
    """
    Map normalised question text -> (category_id, higher_level_category).

    Primary source is the manipulation dataset JSON (it covers every question and is
    checked into the repo). A judged-responses JSON can be passed instead/as well; its
    `full_data` entries carry the same two fields.
    """
    mapping = {}

    if dataset_json and os.path.exists(dataset_json):
        with open(dataset_json, "r") as f:
            categories = json.load(f)["manipulation_categories"]
        for cat in categories:
            for question in cat.get("example_questions", []):
                text = question if isinstance(question, str) else question.get("question_text", "")
                mapping[normalize_text(text)] = (cat["category_id"], cat["higher_level_category"])

    if judge_file and os.path.exists(judge_file):
        with open(judge_file, "r") as f:
            responses = json.load(f)["responses"]
        for resp in responses:
            full = resp["full_data"]
            mapping.setdefault(normalize_text(full["question_text"]),
                               (full["category_id"], full["higher_level_category"]))

    if not mapping:
        raise FileNotFoundError(
            f"No category source found (tried {dataset_json!r} and {judge_file!r})")
    return mapping


# --------------------------------------------------------------------------------------
# reshaping into (rater, unit, value) records
# --------------------------------------------------------------------------------------

def derive_annotation_variables(df, split, metric_name="manipulative"):
    """
    Turn one cleaned split into tidy rows, one per annotator x question, with the
    position labels resolved to model names and the derived preference variables.
    """
    records = []
    for _, row in df.iterrows():
        # position ('A'/'B'/'C') -> model name, as shown to this annotator
        pos_to_model = {}
        for pos in POSITIONS:
            label = row.get(f"response_{pos}_label")
            if pd.isna(label):
                continue
            label = str(label).lower().strip().replace(" ", "_")
            pos_to_model[pos.upper()] = label

        most = row[f"most_{metric_name}_response"]
        least = row[f"least_{metric_name}_response"]
        if pd.isna(most) or pd.isna(least):
            continue
        most, least = str(most).strip().upper(), str(least).strip().upper()
        if most not in pos_to_model or least not in pos_to_model:
            continue

        # 3 = most manipulative, 2 = middle, 1 = least manipulative
        rank = {}
        for pos, model in pos_to_model.items():
            rank[model] = 3 if pos == most else (1 if pos == least else 2)

        def less_manipulative(model_x, model_y):
            """1 if model_x was ranked less manipulative than model_y, else 0."""
            if model_x not in rank or model_y not in rank:
                return np.nan
            return float(rank[model_x] < rank[model_y])

        record = {
            # annotator ids are only unique within a split, so namespace them
            "rater": f"{split}:{row['prolific_id']}",
            "split": split,
            "question_id": f"{split}:{row['question_id']}",
            "question_text": row["question_text"],
            "most_manipulative": pos_to_model[most],
            "least_manipulative": pos_to_model[least],
            "base_vs_suggestive": less_manipulative("base", "suggestive"),
            "steered_vs_suggestive": less_manipulative("suggestive_steered", "suggestive"),
        }
        for pos in POSITIONS:
            model = pos_to_model.get(pos.upper())
            if model is None:
                continue
            record[f"rank__{model}"] = rank[model]
            correctness = row.get(f"correctness_{pos}")
            record[f"correctness__{model}"] = np.nan if pd.isna(correctness) else float(correctness)
        records.append(record)

    return pd.DataFrame(records)


def load_annotations(split_files=None, dataset_json=DEFAULT_DATASET_JSON, judge_file=None,
                     metric_name="manipulative", verbose=True):
    """Load every split, clean it, derive the variables and attach the categories."""
    split_files = split_files or DEFAULT_SPLIT_FILES
    category_map = load_category_map(dataset_json=dataset_json, judge_file=judge_file)

    frames = []
    for split, path in split_files.items():
        if verbose:
            print(f"Loading split {split}: {path}")
        raw = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")
        clean = clean_responses(raw, metric_name=metric_name, verbose=verbose)
        frames.append(derive_annotation_variables(clean, split, metric_name=metric_name))

    df = pd.concat(frames, ignore_index=True)

    lookup = df["question_text"].apply(lambda t: category_map.get(normalize_text(t), (None, None)))
    df["category"] = [c for c, _ in lookup]
    df["higher_level_category"] = [h for _, h in lookup]

    unmatched = df["category"].isna().sum()
    if unmatched and verbose:
        print(f"  ⚠️ {unmatched} annotation rows could not be matched to a category")

    if verbose:
        n_q = df["question_id"].nunique()
        print(f"  -> {len(df)} annotations, {n_q} questions, {df['rater'].nunique()} annotators, "
              f"{df['category'].nunique()} categories")
        counts = df.groupby("question_id").size().value_counts().to_dict()
        print(f"  -> annotators per question: {counts}")
    return df


def to_rating_records(df, variable):
    """
    Long-format (rater, unit, value) triples for one variable.

    Question-level variables give one unit per question; the two response-level variables
    give one unit per question x model, so that a rank/correctness disagreement is scored
    per response rather than being collapsed into a single per-question value.
    """
    level, unit_level = VARIABLES[variable]

    if unit_level == "question":
        sub = df[["rater", "question_id", variable]].dropna(subset=[variable])
        return list(sub["rater"]), list(sub["question_id"]), list(sub[variable]), level

    prefix = {"manipulativeness_rank": "rank__", "correctness": "correctness__"}[variable]
    raters, units, values = [], [], []
    for model in MODELS:
        column = prefix + model
        if column not in df.columns:
            continue
        sub = df[["rater", "question_id", column]].dropna(subset=[column])
        raters += list(sub["rater"])
        units += [f"{q}|{model}" for q in sub["question_id"]]
        values += list(sub[column])
    return raters, units, values, level


# --------------------------------------------------------------------------------------
# agreement computation
# --------------------------------------------------------------------------------------

def compute_agreement(df, variables=None, n_boot=1000, seed=0):
    """Agreement statistics for every variable on the given subset of annotations."""
    variables = variables or list(VARIABLES)
    out = {}
    for variable in variables:
        raters, units, values, level = to_rating_records(df, variable)
        if not values:
            out[variable] = None
            continue
        # categorical values (model names) need an integer coding; the numeric variables
        # are already numeric and must stay so for the ordinal metric
        if isinstance(values[0], str):
            categories = sorted(set(values))
            code = {c: float(i) for i, c in enumerate(categories)}
            values = [code[v] for v in values]
        out[variable] = agreement_report(raters, units, values, level=level,
                                         n_boot=n_boot, seed=seed)
    return out


def compute_all(df, n_boot=1000, seed=0, verbose=True):
    """Agreement combined, per category, per higher-level category and per split."""
    results = {"combined": compute_agreement(df, n_boot=n_boot, seed=seed)}

    for key, column in [("per_category", "category"),
                        ("per_higher_level_category", "higher_level_category"),
                        ("per_split", "split")]:
        group_results = {}
        for value, group in df.groupby(column):
            if verbose:
                print(f"  {column}: {value} ({group['question_id'].nunique()} questions)")
            group_results[str(value)] = compute_agreement(group, n_boot=n_boot, seed=seed)
        results[key] = group_results
    return results


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

def _fmt(value, width=6, decimals=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a".rjust(width)
    return f"{value:{width}.{decimals}f}"


def print_summary_table(results, variables=None, show_ci=True):
    """Krippendorff's alpha per group (rows) and variable (columns)."""
    variables = variables or list(VARIABLES)

    print("\n" + "=" * 100)
    print("KRIPPENDORFF'S ALPHA")
    print("=" * 100)

    header = f"{'group':<34}" + "".join(f"{v[:20]:>21}" for v in variables)
    print(header)
    print("-" * len(header))

    def row(name, stats, n_questions=None):
        label = name if n_questions is None else f"{name} (n={n_questions})"
        line = f"{label[:33]:<34}"
        for variable in variables:
            entry = stats.get(variable)
            if entry is None:
                line += f"{'n/a':>21}"
                continue
            cell = _fmt(entry["alpha"])
            if show_ci and "alpha_ci_low" in entry:
                cell += f" [{_fmt(entry['alpha_ci_low'], 5, 2)},{_fmt(entry['alpha_ci_high'], 5, 2)}]"
            line += f"{cell:>21}"
        print(line)

    combined_n = results["combined"][variables[0]]["n_units"] if results["combined"].get(variables[0]) else None
    row("COMBINED", results["combined"], combined_n)

    for key, title in [("per_split", "Per split"),
                       ("per_higher_level_category", "Per higher-level category"),
                       ("per_category", "Per category")]:
        if key not in results:
            continue
        print("-" * len(header))
        print(f"{title}:")
        for group in sorted(results[key]):
            stats = results[key][group]
            n = stats[variables[0]]["n_units"] if stats.get(variables[0]) else None
            row("  " + group[:24], stats, n)


def print_detail_table(results, variable):
    """Alpha next to Fleiss' kappa, raw percent agreement and the underlying counts."""
    level, unit_level = VARIABLES[variable]
    print("\n" + "=" * 100)
    print(f"DETAIL: {variable}  (level={level}, units={unit_level})")
    print("=" * 100)
    header = (f"{'group':<32}{'alpha':>8}{'95% CI':>18}{'kappa_F':>9}"
              f"{'%agree':>9}{'units':>7}{'ratings':>9}{'raters':>8}")
    print(header)
    print("-" * len(header))

    def row(name, stats):
        entry = stats.get(variable)
        if entry is None:
            print(f"{name:<32}{'n/a':>8}")
            return
        ci = "n/a"
        if "alpha_ci_low" in entry:
            ci = f"[{_fmt(entry['alpha_ci_low'], 5, 2)}, {_fmt(entry['alpha_ci_high'], 5, 2)}]"
        print(f"{name:<32}{_fmt(entry['alpha'])!s:>8}{ci:>18}"
              f"{_fmt(entry['fleiss_kappa'], 8)!s:>9}{_fmt(entry['percent_agreement'], 8)!s:>9}"
              f"{entry['n_units']:>7}{entry['n_ratings']:>9}{entry['n_raters']:>8}")

    row("COMBINED", results["combined"])
    for key, title in [("per_split", "Per split"),
                       ("per_higher_level_category", "Per higher-level category"),
                       ("per_category", "Per category")]:
        if key not in results:
            continue
        print(f"{title}:")
        for group in sorted(results[key]):
            row("  " + group[:30], results[key][group])


def results_to_dataframe(results):
    """Flatten the nested results into a tidy DataFrame (one row per group x variable)."""
    rows = []
    for scope, payload in results.items():
        groups = {"combined": payload} if scope == "combined" else payload
        for group, stats in groups.items():
            for variable, entry in stats.items():
                if entry is None:
                    continue
                rows.append({"scope": scope, "group": group, "variable": variable, **entry})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------

def run(split_files=None, dataset_json=DEFAULT_DATASET_JSON, judge_file=None,
        output=DEFAULT_OUTPUT, n_boot=1000, seed=0, leave_out_cats=None, verbose=True):
    """Load, compute, report and save. Returns (results dict, annotations DataFrame)."""
    df = load_annotations(split_files=split_files, dataset_json=dataset_json,
                          judge_file=judge_file, verbose=verbose)

    if leave_out_cats:
        before = df["question_id"].nunique()
        df = df[~df["category"].isin(leave_out_cats)]
        print(f"\nRestricted to the kept categories: {before} -> {df['question_id'].nunique()} "
              f"questions (dropped {sorted(leave_out_cats)})")

    print("\nComputing agreement statistics"
          f"{f' (bootstrap: {n_boot} resamples)' if n_boot else ''}...")
    results = compute_all(df, n_boot=n_boot, seed=seed, verbose=verbose)

    print_summary_table(results, show_ci=bool(n_boot))
    for variable in ["most_manipulative", "base_vs_suggestive", "correctness"]:
        print_detail_table(results, variable)

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Saved agreement statistics to: {output}")
    return results, df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split_a", default=DEFAULT_SPLIT_FILES["A"])
    parser.add_argument("--split_b", default=DEFAULT_SPLIT_FILES["B"])
    parser.add_argument("--dataset_json", default=DEFAULT_DATASET_JSON,
                        help="source of the question -> category mapping")
    parser.add_argument("--judge_file", default=None,
                        help="optional judged-responses JSON, used as a fallback category source")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--n_boot", type=int, default=1000,
                        help="bootstrap resamples for the alpha CIs (0 disables)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--leave_out_cats", nargs="*", default=None,
                        help="categories to exclude, e.g. --leave_out_cats false_dichotomy "
                             "false_transparency false_causality risk_distortion")
    args = parser.parse_args()

    run(split_files={"A": args.split_a, "B": args.split_b},
        dataset_json=args.dataset_json, judge_file=args.judge_file,
        output=args.output, n_boot=args.n_boot, seed=args.seed,
        leave_out_cats=args.leave_out_cats)


if __name__ == "__main__":
    main()
