"""Validation-only, bounded experiments for the future notebook. Existing notebooks stay intact."""
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import threading
import time

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "4"

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_impact import prepare
from traffic_impact.experiment import (
    FeatureBuilder, FORBIDDEN, WNUM, WCAT, raw_features, preprocessor, score,
    bootstrap_differences,
)

WORK = ROOT / "data/final_notebook_experiments"
RESULT = ROOT / "results/final_notebook_comparison"
LIMITS = {"train": 500010, "validation": 50000, "test": 100000, "stress": 50000}
ALIASES = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
           "CALM": "Calm", "VAR": "Variable", "VARIABLE": "Variable"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def features(raw, after):
    x, groups, _ = raw_features(raw)
    w = pd.to_datetime(raw.Weather_Timestamp, format="mixed", errors="coerce")
    offset = (w - raw._t).dt.total_seconds() / 60
    valid = offset.between(-120, after)
    for c in WNUM:
        x[c] = pd.to_numeric(raw[c], errors="coerce").where(valid)
    for c in WCAT:
        z = raw[c].astype("string")
        if c == "Wind_Direction":
            z = z.str.strip().str.upper().replace(ALIASES)
        x[c] = z.where(valid)
    assert x.loc[~valid, WNUM + WCAT].isna().all().all()
    return x, groups, valid


class Builder(FeatureBuilder):
    def __init__(self, after=60, cap=15):
        self.after, self.cap = after, cap

    def fit(self, raw):
        x, groups, _ = features(raw, self.after)
        self.columns = [c for c in x if c == "Turning_Loop" or x[c].nunique(dropna=False) > 1]
        self.dropped = [c for c in x if c not in self.columns]
        self.groups = {c: groups[c] for c in self.columns}
        self.vocab = {}
        for c in self.columns:
            if pd.api.types.is_string_dtype(x[c].dtype):
                counts = x[c].value_counts()
                kept = counts[counts >= 100].index.tolist()
                if c == "Weather_Condition":
                    kept = kept[:self.cap]
                self.vocab[c] = [str(v) for v in kept if v not in ["Missing", "Other"]]
        assert not set(self.columns) & FORBIDDEN
        return self

    def transform(self, raw):
        x, _, _ = features(raw, self.after)
        for c, vocab in self.vocab.items():
            z = x[c].fillna("Missing").astype(str)
            x[c] = pd.Categorical(z.where(z.isin(vocab + ["Missing"]), "Other"),
                                  categories=vocab + ["Missing", "Other"])
        x = x[self.columns].copy()
        for c in x:
            if c not in self.vocab:
                x[c] = x[c].astype("float32")
        return x[self.columns]


def clean(raw, split, rows):
    raw = raw.copy()
    raw["_t"] = pd.to_datetime(raw.Start_Time, format="mixed", errors="coerce")
    valid = raw._t.notna() & raw.Severity.isin([1, 2, 3, 4])
    rows.append({"split": split, "column": "date_or_label", "invalid": int((~valid).sum())})
    raw = raw.loc[valid].reset_index(drop=True)
    bounds = {"Start_Lat": (-90, 90), "Start_Lng": (-180, 180), "Humidity(%)": (0, 100),
              "Precipitation(in)": (0, np.inf), "Wind_Speed(mph)": (0, np.inf),
              "Visibility(mi)": (0, np.inf), "Pressure(in)": (0, np.inf)}
    for c, (lo, hi) in bounds.items():
        z = pd.to_numeric(raw[c], errors="coerce")
        bad = z.notna() & (~z.between(lo, hi) | ~np.isfinite(z))
        rows.append({"split": split, "column": c, "invalid": int(bad.sum())})
        raw[c] = z.mask(bad)
    return raw


def metrics(raw, p, threshold):
    y = raw.Severity.ge(3).to_numpy(dtype="int8")
    result = score(y, p, raw._priority.to_numpy())
    pred = p >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {**result, "threshold": float(threshold), "precision": precision, "recall": recall,
            "f1": f1, "alert_fraction": float(pred.mean()),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def best_f1(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                   out=np.zeros_like(thresholds), where=precision[:-1] + recall[:-1] > 0)
    return float(thresholds[np.argmax(f1)])


def configurations():
    configs = []
    for n, after, family in itertools.product([200000, 500000], [0, 60], ["hgb", "rf"]):
        params = {"max_iter": 200, "max_leaf_nodes": 31} if family == "hgb" else {
            "n_estimators": 100, "min_samples_leaf": 5, "max_features": "sqrt"}
        configs.append(dict(family=family, train_n=n, after=after, cap=15, stage="core", params=params))
    for leaves, iterations, cap in [(15, 200, 15), (63, 200, 15), (31, 400, 15),
                                    (63, 400, 15), (31, 200, 10), (31, 200, 25)]:
        configs.append(dict(family="hgb", train_n=500000, after=60, cap=cap, stage="tuning",
                            params={"max_iter": iterations, "max_leaf_nodes": leaves}))
    for leaf in [10, 20]:
        configs.append(dict(family="rf", train_n=500000, after=60, cap=15, stage="tuning",
                            params={"n_estimators": 100, "min_samples_leaf": leaf, "max_features": "sqrt"}))
    for index, c in enumerate(configs):
        c["name"] = f"e{index+1:02d}_{c['family']}_{c['train_n']}_w{c['after']}_c{c['cap']}"
    return configs


class MemoryWatch:
    def __enter__(self):
        self.stop = threading.Event()
        self.peak = psutil.Process().memory_info().rss
        def watch():
            while not self.stop.wait(.1):
                self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def self_check():
    base = pd.read_pickle(ROOT / "data/main_notebook/data/cache/train.pkl").iloc[:8].copy()
    base["State"] = ["CA", "TX"] * 4
    offsets = [-121, -120, -30, 0, 60, 61, np.nan, 30]
    base["Weather_Timestamp"] = base._t + pd.to_timedelta(offsets, unit="min")
    assert features(base, 60)[2].tolist() == [False, True, True, True, True, False, False, True]
    assert features(base, 0)[2].tolist() == [False, True, True, True, False, False, False, False]
    b = Builder().fit(base)
    assert "Turning_Loop" in b.columns and not set(b.columns) & FORBIDDEN
    changed = base.copy()
    changed["State"] = "NEW_UNKNOWN_STATE"
    before = json.dumps(b.vocab, sort_keys=True)
    assert b.transform(changed).State.eq("Other").all()
    assert before == json.dumps(b.vocab, sort_keys=True)
    print("Boundary, missing-weather, retained-constant and frozen-vocabulary checks passed.", flush=True)


def run():
    started = time.perf_counter()
    WORK.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    (WORK / "models").mkdir(exist_ok=True)
    notebooks = [ROOT / "notebooks/MAIN_NoteBook.ipynb", ROOT.parent / "01_preprocessing_kamar.ipynb",
                 ROOT.parent / "02_models_kamar.ipynb"]
    original_hashes = {str(p): sha(p) for p in notebooks}
    self_check()
    if "--check-only" in sys.argv:
        return
    configs = configurations()
    csv = Path(json.loads((ROOT / "results/audit/audit.json").read_text())["file"])
    signature = {"csv": str(csv), "bytes": csv.stat().st_size, "mtime_ns": csv.stat().st_mtime_ns,
                 "prepare_sha256": sha(prepare.__file__), "pandas": pd.__version__,
                 "numpy": np.__version__, "limits": LIMITS}
    dump(RESULT / "protocol.json", {"configs": configs, "versions": {"python": sys.version,
         "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
         "script_sha256": sha(__file__), "signature": signature, "notebook_hashes_before": original_hashes,
         "selection": "Highest validation AP within the +60-minute scenario, independently per family. Exact ties: faster fit.",
         "primary_metric": "Average Precision", "threads": 4, "seed": 42,
         "random_forest_representation": "dense float32 after training-only median imputation and one-hot",
         "threshold": "Maximum F1 on validation only; reused unchanged on historical years.",
         "test_used_for_selection": False, "bootstrap_note": "200 paired weekly resamples; descriptive on reused validation."})
    stamp = WORK / "cache_manifest.json"
    cached = json.loads(stamp.read_text()) if stamp.exists() else {}
    paths = {s: WORK / "data/cache" / f"{s}.pkl" for s in LIMITS}
    if cached.get("signature") != signature or not all(p.exists() and sha(p) == cached.get("sha", {}).get(s) for s, p in paths.items()):
        prepare.run(csv, out=WORK, limits=LIMITS)
        dump(stamp, {"signature": signature, "sha": {s: sha(p) for s, p in paths.items()}})
    corrections = []
    raw = {s: clean(pd.read_pickle(p), s, corrections) for s, p in paths.items()}
    raw["train"] = raw["train"].nsmallest(500000, "_priority").sort_values("_t").reset_index(drop=True)
    assert len(raw["train"]) == 500000
    for s, years in {"train": range(2016, 2020), "validation": [2020], "test": [2021], "stress": [2022]}.items():
        assert raw[s].Source.eq("Source2").all() and raw[s]._t.dt.year.isin(years).all()
        assert raw[s]._event.is_unique
    for a, b in itertools.combinations(raw, 2):
        assert not np.intersect1d(raw[a]._event, raw[b]._event).size
        assert raw[a]._t.max() < raw[b]._t.min()
    historic_val = pd.read_pickle(ROOT / "data/main_notebook/data/cache/validation.pkl")
    assert np.array_equal(raw["validation"]._event, historic_val._event)
    del historic_val
    samples, weather_rows = [], []
    for s, d in raw.items():
        samples.append({"split": s, "n": len(d), "positive_rate": float(d.Severity.ge(3).mean()),
                        "first": str(d._t.min()), "last": str(d._t.max())})
        for after in [0, 60]:
            _, _, valid = features(d, after)
            weather_rows.append({"split": s, "after_minutes": after, "events": len(d),
                                 "eligible": int(valid.sum()), "eligible_fraction": float(valid.mean())})
    pd.DataFrame(samples).to_csv(RESULT / "samples.csv", index=False)
    pd.DataFrame(corrections).to_csv(RESULT / "cleaning.csv", index=False)
    pd.DataFrame(weather_rows).to_csv(RESULT / "weather.csv", index=False)
    rows, preds = [], {}
    yv = raw["validation"].Severity.ge(3).to_numpy(dtype="int8")
    for c in configs:
        name = c["name"]
        prediction_path = WORK / f"{name}_validation.npy"
        model_path = WORK / "models" / f"{name}.joblib"
        row_path = WORK / f"{name}.json"
        if row_path.exists() and model_path.exists() and prediction_path.exists():
            saved = json.loads(row_path.read_text())
            if saved.get("config") == c and saved.get("script_sha256") == sha(__file__):
                rows.append(saved["row"])
                preds[name] = np.load(prediction_path)
                continue
        print(f"Starting {name}: {c['params']}", flush=True)
        gc.collect()
        train = raw["train"].nsmallest(c["train_n"], "_priority").sort_values("_t").reset_index(drop=True)
        assert len(train) == c["train_n"]
        with MemoryWatch() as memory:
            t = time.perf_counter()
            builder = Builder(c["after"], c["cap"]).fit(train)
            x, xv = builder.transform(train), builder.transform(raw["validation"])
            yt = train.Severity.ge(3).to_numpy(dtype="int8")
            assert len(builder.columns) == 34
            pre = None
            if c["family"] == "rf":
                pre = preprocessor(x)
                x = pre.fit_transform(x).astype("float32").toarray()
                xv = pre.transform(xv).astype("float32").toarray()
                model = RandomForestClassifier(n_jobs=4, random_state=42, **c["params"])
            else:
                model = HistGradientBoostingClassifier(learning_rate=.1, early_stopping=False,
                    categorical_features="from_dtype", random_state=42, **c["params"])
            preparation_seconds = time.perf_counter() - t
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
            row.update({"params": json.dumps(c["params"], sort_keys=True),
                "input_columns": len(builder.columns), "encoded_columns": x.shape[1],
                "preparation_seconds": preparation_seconds, "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds, **metrics(raw["validation"], p, threshold)})
        row["peak_process_mb"] = memory.peak / 1e6
        joblib.dump({"builder": builder, "pre": pre, "model": model, "threshold": threshold}, model_path, compress=3)
        np.save(prediction_path, p)
        dump(row_path, {"config": c, "script_sha256": sha(__file__), "row": row})
        rows.append(row)
        preds[name] = p
        pd.DataFrame(rows).to_csv(RESULT / "validation.csv", index=False)
        print(f"Finished {name}: AP={row['ap']:.6f}, AUC={row['auc']:.6f}, F1={row['f1']:.6f}, fit={fit_seconds:.1f}s, peak={row['peak_process_mb']:.0f}MB", flush=True)
        del train, builder, x, xv, yt, pre, model
        gc.collect()
    table = pd.DataFrame(rows)
    winners = table[table.after.eq(60)].sort_values(["ap", "fit_seconds"], ascending=[False, True]).groupby("family", sort=False).head(1)
    dump(RESULT / "frozen_selection.json", {"selected": winners.to_dict("records"),
        "test_used_for_selection": False, "note": "Best among tested configurations; not a global optimum."})
    intervals = []
    core = table[table.stage.eq("core")]
    for family in ["hgb", "rf"]:
        for after in [0, 60]:
            pair = core[core.family.eq(family) & core.after.eq(after)].sort_values("train_n")
            low, high = pair.name.tolist()
            intervals.extend(bootstrap_differences(raw["validation"], yv, {low: preds[low], high: preds[high]}, reference=high, B=200))
        pair = core[core.family.eq(family) & core.train_n.eq(500000)].sort_values("after")
        strict, delayed = pair.name.tolist()
        intervals.extend(bootstrap_differences(raw["validation"], yv, {strict: preds[strict], delayed: preds[delayed]}, reference=delayed, B=200))
    pd.DataFrame(intervals).to_csv(RESULT / "paired_validation_intervals.csv", index=False)
    evaluation = []
    for _, winner in winners.iterrows():
        fitted = joblib.load(WORK / "models" / f"{winner['name']}.joblib")
        for split in ["test", "stress"]:
            z = fitted["builder"].transform(raw[split])
            if fitted["pre"] is not None:
                z = fitted["pre"].transform(z).astype("float32").toarray()
            with threadpool_limits(limits=4):
                p = fitted["model"].predict_proba(z)[:, 1]
            evaluation.append({"name": winner["name"], "family": winner["family"], "split": split,
                               **metrics(raw[split], p, fitted["threshold"])})
            np.save(WORK / f"{winner['name']}_{split}.npy", p)
        del fitted, z
        gc.collect()
    prior = float(raw["train"].Severity.ge(3).mean())
    for split in ["validation", "test", "stress"]:
        evaluation.append({"name": "constant", "family": "prior", "split": split,
                           **metrics(raw[split], np.full(len(raw[split]), prior), prior)})
    pd.DataFrame(evaluation).to_csv(RESULT / "historical_evaluation.csv", index=False)
    final_hashes = {str(p): sha(p) for p in notebooks}
    dump(RESULT / "run_summary.json", {"seconds": time.perf_counter() - started,
         "notebooks_unchanged": final_hashes == original_hashes, "notebook_hashes_after": final_hashes,
         "completed_fits": len(rows), "finished_at": pd.Timestamp.now().isoformat(),
         "memory_note": "Peak total RSS of this process, sampled every 0.1s during preparation/fit/prediction. Not incremental estimator memory."})
    print("Complete. Selected models:", flush=True)
    print(winners[["name", "ap", "auc", "f1", "fit_seconds"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
