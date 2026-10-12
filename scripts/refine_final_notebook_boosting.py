"""Compare both volumes and three weather caps using validation-selected tree settings."""
import gc
import itertools
import json
from pathlib import Path
import time

from compare_final_notebook_options import (
    ROOT, RESULT, WORK, Builder, MemoryWatch, clean, dump, sha, best_f1, metrics,
    HistGradientBoostingClassifier, threadpool_limits, np, pd, joblib,
    bootstrap_differences,
)


def main():
    started = time.perf_counter()
    base_summary = json.loads((RESULT / "run_summary.json").read_text())
    table = pd.read_csv(RESULT / "validation.csv")
    initial = table[table.stage.ne("followup")]
    assert len(initial) == 16
    best = initial[initial.family.eq("hgb") & initial.after.eq(60)].sort_values(
        ["ap", "fit_seconds"], ascending=[False, True]).iloc[0]
    params = json.loads(best.params)
    configs = []
    for n, cap in itertools.product([200000, 500000], [10, 15, 25]):
        if n == int(best.train_n) and cap == int(best.cap):
            continue
        name = f"e{17+len(configs):02d}_hgb_{n}_w60_c{cap}_refined"
        configs.append({"name": name, "family": "hgb", "train_n": n, "after": 60,
                        "cap": cap, "stage": "followup", "params": params})
    dump(RESULT / "followup_protocol.json", {"script_sha256": sha(__file__), "configs": configs,
        "tree_settings_from": best["name"], "tree_settings_selected_on": "Validation 2020 AP only",
        "selection": "Update HGB selection by validation AP across all candidates in the +60-minute scenario; exact ties: faster fit.",
        "reason": "Tree-capacity gain exceeded the fixed-settings volume gain; compare volume and caps at the selected tree capacity.",
        "test_used_for_selection": False})
    if not (RESULT / "base_frozen_selection.json").exists():
        dump(RESULT / "base_frozen_selection.json", json.loads((RESULT / "frozen_selection.json").read_text()))
    notebooks = list(base_summary["notebook_hashes_after"])
    before = {p: sha(p) for p in notebooks}
    corrections = []
    raw = {s: clean(pd.read_pickle(WORK / "data/cache" / f"{s}.pkl"), s, corrections)
           for s in ["train", "validation", "test", "stress"]}
    raw["train"] = raw["train"].nsmallest(500000, "_priority").sort_values("_t").reset_index(drop=True)
    rows = initial.to_dict("records")
    yv = raw["validation"].Severity.ge(3).to_numpy(dtype="int8")
    for c in configs:
        name = c["name"]
        row_path = WORK / f"{name}.json"
        model_path = WORK / "models" / f"{name}.joblib"
        prediction_path = WORK / f"{name}_validation.npy"
        if row_path.exists() and model_path.exists() and prediction_path.exists():
            saved = json.loads(row_path.read_text())
            if saved.get("config") == c and saved.get("script_sha256") == sha(__file__):
                rows.append(saved["row"])
                continue
        print(f"Starting {name}: {params}", flush=True)
        gc.collect()
        train = raw["train"].nsmallest(c["train_n"], "_priority").sort_values("_t").reset_index(drop=True)
        with MemoryWatch() as memory:
            t = time.perf_counter()
            builder = Builder(60, c["cap"]).fit(train)
            x, xv = builder.transform(train), builder.transform(raw["validation"])
            yt = train.Severity.ge(3).to_numpy(dtype="int8")
            prep_seconds = time.perf_counter() - t
            model = HistGradientBoostingClassifier(learning_rate=.1, early_stopping=False,
                categorical_features="from_dtype", random_state=42, **params)
            t = time.perf_counter()
            with threadpool_limits(limits=4):
                model.fit(x, yt)
            fit_seconds = time.perf_counter() - t
            t = time.perf_counter()
            with threadpool_limits(limits=4):
                p = model.predict_proba(xv)[:, 1]
            predict_seconds = time.perf_counter() - t
            threshold = best_f1(yv, p)
            row = {k: v for k, v in c.items() if k != "params"}
            row.update({"params": json.dumps(params, sort_keys=True), "input_columns": len(builder.columns),
                "encoded_columns": x.shape[1], "preparation_seconds": prep_seconds,
                "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                **metrics(raw["validation"], p, threshold)})
        row["peak_process_mb"] = memory.peak / 1e6
        joblib.dump({"builder": builder, "pre": None, "model": model, "threshold": threshold}, model_path, compress=3)
        np.save(prediction_path, p)
        dump(row_path, {"config": c, "script_sha256": sha(__file__), "row": row})
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULT / "validation.csv", index=False)
        print(f"Finished {name}: AP={row['ap']:.6f}, F1={row['f1']:.6f}, fit={fit_seconds:.1f}s", flush=True)
        del train, builder, x, xv, yt, model
    table = pd.DataFrame(rows)
    selected = table[table.after.eq(60)].sort_values(["ap", "fit_seconds"], ascending=[False, True]).groupby("family", sort=False).head(1)
    dump(RESULT / "frozen_selection.json", {"selected": selected.to_dict("records"),
        "test_used_for_selection": False, "note": "Best among 21 tested configurations, including validation-motivated followup; not a global optimum."})
    refined = pd.concat([table[table.stage.eq("followup")], table[table.name.eq(best["name"])]])
    interval_rows = []
    for cap in [10, 15, 25]:
        pair = refined[refined.cap.eq(cap)].sort_values("train_n")
        low, high = pair.name.tolist()
        predictions = {n: np.load(WORK / f"{n}_validation.npy") for n in [low, high]}
        interval_rows.extend(bootstrap_differences(raw["validation"], yv, predictions, reference=high, B=200))
    pd.DataFrame(interval_rows).to_csv(RESULT / "refined_volume_intervals.csv", index=False)
    hgb_best = selected[selected.family.eq("hgb")].iloc[0]
    fitted = joblib.load(WORK / "models" / f"{hgb_best['name']}.joblib")
    existing = pd.read_csv(RESULT / "historical_evaluation.csv")
    evaluation = existing[existing.family.ne("hgb")].to_dict("records")
    for split in ["test", "stress"]:
        x = fitted["builder"].transform(raw[split])
        with threadpool_limits(limits=4):
            p = fitted["model"].predict_proba(x)[:, 1]
        np.save(WORK / f"{hgb_best['name']}_{split}.npy", p)
        evaluation.append({"name": hgb_best["name"], "family": "hgb", "split": split,
                           **metrics(raw[split], p, fitted["threshold"])})
    pd.DataFrame(evaluation).to_csv(RESULT / "historical_evaluation.csv", index=False)
    after = {p: sha(p) for p in notebooks}
    followup_seconds = time.perf_counter() - started
    base_seconds = base_summary.get("base_series_seconds", base_summary["seconds"])
    dump(RESULT / "run_summary.json", {**base_summary, "base_series_seconds": base_seconds,
        "followup_seconds": followup_seconds, "seconds": base_seconds + followup_seconds,
        "completed_fits": len(table), "notebooks_unchanged": base_summary["notebooks_unchanged"] and before == after == base_summary["notebook_hashes_after"],
        "notebook_hashes_after": after, "finished_at": pd.Timestamp.now().isoformat()})
    print("Complete. Validation-selected models:", flush=True)
    print(selected[["name", "ap", "auc", "f1", "fit_seconds"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
