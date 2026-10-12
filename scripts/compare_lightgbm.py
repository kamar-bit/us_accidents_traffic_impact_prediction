"""Compare LightGBM and XGBoost with the final notebook's HistGradientBoosting model.

The script runs the final notebook's own preparation cells (samples, cleaning,
weather rule, 33 inputs, training vocabularies) and its boosting cell, then
trains LightGBM and XGBoost on exactly the same matrices. Thresholds are chosen on 2020
with the notebook's F1 rule and kept fixed for 2021 and 2022. Nothing in the
notebook is modified; outputs go to results/lightgbm_comparison/.

Usage, from the project root:

    python scripts/compare_lightgbm.py [--csv path/to/US_Accidents_March23.csv]
"""
import argparse
import ast
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/FINAL_NOTEBOOK.ipynb"
OUTPUT = ROOT / "results/lightgbm_comparison"
FOREST_MARKER = "RandomForestClassifier(n_estimators"
EVALUATION_FUNCTIONS = {"evaluate", "f1_threshold"}


def notebook_code():
    """Return the preparation and boosting cells, and the evaluation functions."""
    cells = [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells if c.cell_type == "code"]
    stop = next(i for i, source in enumerate(cells) if FOREST_MARKER in source)
    functions = []
    for source in cells[stop:]:
        functions += [node for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef) and node.name in EVALUATION_FUNCTIONS]
    assert {f.name for f in functions} == EVALUATION_FUNCTIONS, "Notebook evaluation functions not found"
    assert any("HistGradientBoostingClassifier(**boosting_settings)" in s for s in cells[:stop])
    return cells[:stop], ast.Module(body=functions, type_ignores=[])


def lightgbm_settings(seed):
    """Match the notebook boosting configuration as closely as LightGBM allows.

    63 leaves, 400 rounds, learning rate 0.1, 255 bins, at least 20 events per
    leaf, no L2 penalty, no row or column subsampling and no early stopping,
    as in HistGradientBoostingClassifier's defaults used by the notebook.
    """
    return {"n_estimators": 400, "num_leaves": 63, "learning_rate": 0.1, "max_depth": -1,
            "min_child_samples": 20, "min_child_weight": 1e-3, "reg_lambda": 0.0,
            "max_bin": 255, "subsample": 1.0, "colsample_bytree": 1.0,
            "random_state": seed, "n_jobs": 4, "verbose": -1}


def xgboost_settings(seed):
    """Match the notebook boosting configuration with XGBoost's histogram trees.

    Leaf-wise growth capped at 63 leaves (no depth limit), 400 rounds, learning
    rate 0.1, 255 bins, native categories, no L2 penalty, no subsampling and no
    early stopping. XGBoost has no minimum event count per leaf; its
    min_child_weight (hessian sum) keeps the default of 1.
    """
    return {"n_estimators": 400, "tree_method": "hist", "grow_policy": "lossguide",
            "max_leaves": 63, "max_depth": 0, "learning_rate": 0.1, "max_bin": 255,
            "min_child_weight": 1.0, "reg_lambda": 0.0, "subsample": 1.0,
            "colsample_bytree": 1.0, "enable_categorical": True,
            "objective": "binary:logistic", "random_state": seed, "n_jobs": 4}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, help="US_Accidents_March23.csv (default: notebook lookup)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    os.chdir(ROOT)
    preparation, functions = notebook_code()
    namespace = {"__name__": "final_notebook"}
    quiet = {"display": lambda *a, **k: None}
    for index, source in enumerate(preparation):
        if args.csv is not None and "CSV_PATH = find_csv()" in source:
            source = source.replace("CSV_PATH = find_csv()", f"CSV_PATH = Path({str(args.csv.resolve())!r})")
        exec(compile(source, f"final_notebook_cell_{index}", "exec"), namespace)
        namespace.update(quiet)
        plt.close("all")
        print(f"Notebook cell {index + 1}/{len(preparation)} done", flush=True)
    exec(compile(functions, "final_notebook_evaluation", "exec"), namespace)
    capacity = namespace["capacity"]
    main_settings = namespace["boosting_settings"]
    boosting_seconds = float(capacity.loc[capacity["Maximum leaves"].eq(main_settings["max_leaf_nodes"])
        & capacity["Iterations"].eq(main_settings["max_iter"]), "Fit seconds"].iloc[0])

    X, y, raw_clean, SEED = namespace["X"], namespace["y"], namespace["raw_clean"], namespace["SEED"]
    boosting, evaluate, f1_threshold = namespace["boosting"], namespace["evaluate"], namespace["f1_threshold"]
    threadpool_limits = namespace["threadpool_limits"]

    settings = lightgbm_settings(SEED)
    lightgbm = LGBMClassifier(**settings)
    start = time.perf_counter()
    with threadpool_limits(limits=4):
        lightgbm.fit(X["train"], y["train"], categorical_feature="auto")
    lightgbm_seconds = time.perf_counter() - start

    xgb_settings = xgboost_settings(SEED)
    xgboost = XGBClassifier(**xgb_settings)
    start = time.perf_counter()
    with threadpool_limits(limits=4):
        xgboost.fit(X["train"], y["train"])
    xgboost_seconds = time.perf_counter() - start

    models = [("Boosting", boosting), ("LightGBM", lightgbm), ("XGBoost", xgboost)]
    prior = float(y["train"].mean())
    predictions, prediction_seconds = {}, {name: 0.0 for name, _ in models}
    with threadpool_limits(limits=4):
        for split in ["validation", "test", "stress"]:
            predictions[split] = {"Constant": np.full(len(y[split]), prior)}
            for name, model in models:
                start = time.perf_counter()
                predictions[split][name] = model.predict_proba(X[split])[:, 1]
                prediction_seconds[name] += time.perf_counter() - start
    thresholds = {name: f1_threshold(y["validation"], values)
                  for name, values in predictions["validation"].items()}
    scores = pd.DataFrame([{"Period": split, "Model": name,
        **evaluate(y[split], values, thresholds[name], raw_clean[split]._priority.to_numpy())}
        for split in predictions for name, values in predictions[split].items()])

    args.output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output / "scores.csv", index=False)
    periods = {"validation": "2020 (validation)", "test": "2021 (historical)", "stress": "2022 to Sep 8"}
    table = scores.assign(Period=scores.Period.map(periods))[
        ["Period", "Model", "AP", "ROC-AUC", "Precision", "Recall", "F1", "Precision top 10%", "Recall top 10%"]]

    ap = scores.pivot(index="Period", columns="Model", values="AP").reindex(list(periods))
    differences = pd.DataFrame({"LightGBM - Boosting": ap.LightGBM - ap.Boosting,
                                "XGBoost - Boosting": ap.XGBoost - ap.Boosting}).rename(index=periods)

    frozen = json.loads((ROOT / "results/final_notebook_comparison/frozen_selection.json").read_text())
    reference = next(r for r in frozen["selected"] if r["family"] == "hgb")
    boosting_validation = scores[(scores.Period == "validation") & (scores.Model == "Boosting")].iloc[0]

    summary = {"lightgbm_settings": settings, "lightgbm_trees": int(lightgbm.booster_.num_trees()),
        "xgboost_settings": xgb_settings, "xgboost_trees": int(xgboost.get_booster().num_boosted_rounds()),
        "boosting_settings": namespace["boosting_settings"], "boosting_iterations": int(boosting.n_iter_),
        "fit_seconds": {"Boosting": round(boosting_seconds, 1), "LightGBM": round(lightgbm_seconds, 1),
                        "XGBoost": round(xgboost_seconds, 1)},
        "predict_seconds_all_periods": {k: round(v, 2) for k, v in prediction_seconds.items()},
        "thresholds_from_2020": thresholds,
        "lightgbm_minus_boosting_ap": differences["LightGBM - Boosting"].round(4).to_dict(),
        "xgboost_minus_boosting_ap": differences["XGBoost - Boosting"].round(4).to_dict(),
        "boosting_reproduces_recorded_validation_ap": bool(
            np.isclose(boosting_validation.AP, reference["ap"], atol=1e-8, rtol=0)),
        "training_events": int(len(y["train"])), "logical_inputs": int(X["train"].shape[1])}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    from sklearn.metrics import precision_recall_curve
    for name in ["Boosting", "LightGBM", "XGBoost"]:
        precision, recall, _ = precision_recall_curve(y["test"], predictions["test"][name])
        axes[0].plot(recall, precision, label=name)
    axes[0].axhline(y["test"].mean(), color="gray", linestyle="--", label="Positive share")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Historical 2021 evaluation", ylim=(0, 1.02))
    axes[0].legend()
    ranking = ap.copy()
    ranking.index = ["2020", "2021", "2022 partial"]
    ranking[["Constant", "Boosting", "LightGBM", "XGBoost"]].plot.bar(ax=axes[1], rot=0)
    axes[1].set(ylabel="Average precision", xlabel="", title="Ranking across periods", ylim=(0, 1))
    axes[1].legend(loc="upper left", ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output / "boosting_vs_lightgbm.png")

    print("\n" + table.round(3).to_string(index=False))
    print("\nAP difference against HistGradientBoosting:\n" + differences.round(4).to_string())
    print(f"\nFit time: HistGradientBoosting {boosting_seconds:.0f}s, LightGBM {lightgbm_seconds:.0f}s, "
          f"XGBoost {xgboost_seconds:.0f}s")
    print(f"Outputs written to {args.output}")


if __name__ == "__main__":
    main()
