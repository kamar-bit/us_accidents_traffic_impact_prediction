"""Predeclared comparisons, frozen selection, then held-out evaluation.

Usage: python -m traffic_impact.experiment [--out DIRECTORY] [--bootstrap 500]
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from threadpoolctl import threadpool_limits

from .paths import ROOT, artifacts_dir, cache_dir
from .prepare import save_json

POI = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
       "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]
WNUM = ["Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
        "Wind_Speed(mph)", "Precipitation(in)"]
WCAT = ["Wind_Direction", "Weather_Condition"]
LIGHT = ["Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]
GROUPS = ["time", "geography", "road", "light", "weather"]
FORBIDDEN = {"ID", "Source", "Severity", "End_Time", "End_Lat", "End_Lng", "Distance(mi)",
             "duration_min", "Description", "Street", "City", "County", "Zipcode", "Airport_Code"}


def raw_features(raw):
    t = raw._t
    x = pd.DataFrame(index=raw.index)
    x["hour"] = t.dt.hour
    x["day_of_week"] = t.dt.dayofweek
    x["month"] = t.dt.month
    x["weekend"] = (t.dt.dayofweek >= 5).astype(int)
    groups = {c: "time" for c in x}
    for c in ["Start_Lat", "Start_Lng", "State", "Timezone"]:
        x[c] = raw[c]
        groups[c] = "geography"
    for c in POI:
        x[c] = raw[c].astype(float)
        groups[c] = "road"
    for c in LIGHT:
        x[c] = raw[c].astype("string")
        groups[c] = "light"
    weather_time = pd.to_datetime(raw.Weather_Timestamp, format="mixed", errors="coerce")
    lag = (t - weather_time).dt.total_seconds() / 60
    valid = lag.between(0, 120)
    for c in WNUM:
        x[c] = pd.to_numeric(raw[c], errors="coerce").where(valid)
        groups[c] = "weather"
    for c in WCAT:
        z = raw[c].astype("string")
        if c == "Wind_Direction":
            z = z.str.strip().str.upper().replace({"NORTH": "N", "SOUTH": "S", "EAST": "E",
                "WEST": "W", "CALM": "Calm", "VAR": "Variable", "VARIABLE": "Variable"})
        x[c] = z.where(valid)
        groups[c] = "weather"
    return x, groups, valid


class FeatureBuilder:
    """Only fit() may learn category vocabularies or remove constants."""
    def fit(self, raw):
        x, groups, _ = raw_features(raw)
        self.columns = [c for c in x if x[c].nunique(dropna=False) > 1]
        self.dropped = [c for c in x if c not in self.columns]
        self.groups = {c: groups[c] for c in self.columns}
        self.vocab = {}
        for c in self.columns:
            if pd.api.types.is_string_dtype(x[c].dtype):
                freq = x[c].value_counts()
                top = freq[freq >= 100].index.tolist()
                if c == "Weather_Condition":
                    top = top[:15]
                self.vocab[c] = [str(v) for v in top if v not in ["Missing", "Other"]]
        assert not (set(self.columns) & FORBIDDEN)
        return self

    def transform(self, raw):
        x, _, _ = raw_features(raw)
        for c, vocab in self.vocab.items():
            z = x[c].fillna("Missing").astype(str)
            z = z.where(z.isin(vocab + ["Missing"]), "Other")
            x[c] = pd.Categorical(z, categories=vocab + ["Missing", "Other"])
        return x[self.columns]


def missing_features(x):
    return pd.DataFrame({c: (x[c].eq("Missing") if str(x[c].dtype) == "category" else x[c].isna()).astype(int)
                         for c in x}, index=x.index)


def score(y, p, tie):
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(np.ceil(len(y) * .1)))
    selected = np.lexsort((tie, -p))[:k]
    positives = int(y.sum())
    prevalence = float(y.mean())
    precision10 = float(y[selected].mean())
    return {"n": len(y), "prevalence": float(y.mean()),
            "ap": float(average_precision_score(y, p)) if positives else None,
            "auc": float(roc_auc_score(y, p)) if 0 < positives < len(y) else None,
            "recall10": float(y[selected].sum()/positives) if positives else None,
            "precision10": precision10,
            "lift10": precision10 / prevalence if prevalence else None,
            "brier": float(brier_score_loss(y, p))}


def format_score(label, metrics):
    def value(key, percent=False):
        metric = metrics.get(key)
        if metric is None or pd.isna(metric):
            return "n/a"
        return f"{metric:.1%}" if percent else f"{metric:.4f}"

    lift = metrics.get("lift10")
    lift_text = "n/a" if lift is None or pd.isna(lift) else f"{lift:.2f}x"
    return (f"{label:<26} AP {value('ap')} | AUC {value('auc')} | "
            f"P@10 {value('precision10', True)} | R@10 {value('recall10', True)} | "
            f"Lift@10 {lift_text}")


class OneRule:
    def fit(self, x, y, column):
        self.column = column
        self.prior = float(np.mean(y))
        self.categorical = str(x.dtype) == "category"
        y = np.asarray(y)
        if self.categorical:
            tab = pd.DataFrame({"x": x.astype(str).to_numpy(), "y": y}).groupby("x").y.agg(["sum", "count"])
            self.mapping = ((tab["sum"]+50*self.prior)/(tab["count"]+50)).to_dict()
        else:
            self.median = float(x.median()) if x.notna().any() else 0.
            values = x.fillna(self.median).to_numpy(dtype=float)
            order = np.argsort(values, kind="stable")
            v, sy = values[order], y[order]
            left_n = np.arange(1, len(y))
            left_pos = np.cumsum(sy)[:-1]
            min_n = int(np.ceil(.01*len(y)))
            ok = (v[:-1] < v[1:]) & (left_n >= min_n) & (len(y)-left_n >= min_n)
            indices = np.flatnonzero(ok)
            if len(indices):
                nl, pl = left_n[indices], left_pos[indices]
                nr, pr = len(y)-nl, y.sum()-pl
                rl, rr = pl/nl, pr/nr
                top_pos = np.where(rl >= rr, pl, pr)
                top_rate = np.maximum(rl, rr)
                aps = top_pos/y.sum()*top_rate + (y.sum()-top_pos)/y.sum()*self.prior
                i = indices[np.argmax(aps)]
                self.threshold = float(v[i] + (v[i+1]-v[i])/2)
                nl, pl = i+1, left_pos[i]
                self.left = float((pl+50*self.prior)/(nl+50))
                self.right = float((y.sum()-pl+50*self.prior)/(len(y)-nl+50))
            else:
                self.threshold, self.left, self.right = 0., self.prior, self.prior
        return self

    def predict(self, x):
        if self.categorical:
            return x.astype(str).map(self.mapping).fillna(self.prior).to_numpy(dtype=float)
        return np.where(x.fillna(self.median).to_numpy() <= self.threshold, self.left, self.right)

    def describe(self):
        if self.categorical:
            return {"variable": self.column, "kind": "smoothed_category_rate", "rates": self.mapping,
                    "unknown_rate": self.prior, "smoothing": 50}
        return {"variable": self.column, "kind": "threshold", "threshold": self.threshold,
                "p_below_or_equal": self.left, "p_above": self.right, "missing_imputed": self.median}


def preprocessor(x, scale=False):
    cats = [c for c in x if str(x[c].dtype) == "category"]
    nums = [c for c in x if c not in cats]
    numeric = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()) if scale else SimpleImputer(strategy="median", keep_empty_features=True)
    return ColumnTransformer([("num", numeric, nums), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cats)],
                             sparse_threshold=1.0, verbose_feature_names_out=False)


class Model:
    def __init__(self, kind, columns, parameters=None):
        self.kind, self.columns, self.parameters = kind, list(columns), parameters or {}

    def fit(self, x, y):
        z = x[self.columns]
        if self.kind == "prior":
            self.prior = float(np.mean(y))
        elif self.kind == "one":
            self.fitted = OneRule().fit(z.iloc[:, 0], y, self.columns[0])
        elif self.kind in ["tree", "logistic"]:
            self.pre = preprocessor(z, self.kind == "logistic")
            encoded = self.pre.fit_transform(z)
            estimator = (DecisionTreeClassifier(max_leaf_nodes=4, max_depth=3, min_samples_leaf=.01, random_state=42)
                         if self.kind == "tree" else LogisticRegression(C=self.parameters.get("C", 1), max_iter=1000, random_state=42))
            self.fitted = estimator.fit(encoded, y)
        elif self.kind == "hgb":
            self.fitted = HistGradientBoostingClassifier(learning_rate=.1, categorical_features="from_dtype",
                early_stopping=False, random_state=42, **self.parameters).fit(z, y)
        else:
            raise ValueError(self.kind)
        return self

    def predict(self, x):
        z = x[self.columns]
        if self.kind == "prior":
            return np.full(len(x), self.prior)
        if self.kind == "one":
            return self.fitted.predict(z.iloc[:, 0])
        if self.kind in ["tree", "logistic"]:
            z = self.pre.transform(z)
        return self.fitted.predict_proba(z)[:, 1]

    def describe(self):
        d = {"kind": self.kind, "columns": self.columns, "parameters": self.parameters}
        if self.kind == "one":
            d["rule"] = self.fitted.describe()
        if self.kind == "tree":
            d["tree"] = export_text(self.fitted, feature_names=list(self.pre.get_feature_names_out()), decimals=4,
                                     show_weights=True)
        return d


class RankedMetrics:
    """Pre-sort once; evaluate weekly bootstrap multiplicities without sorting again."""
    def __init__(self, y, p, tie):
        self.order = np.lexsort((tie, -p))
        self.y = np.asarray(y)[self.order]
        sorted_p = np.asarray(p)[self.order]
        self.starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1])+1]

    def weighted(self, weights):
        w = weights[self.order]
        n = w.sum()
        positives = np.dot(w, self.y)
        if not positives:
            return np.array([np.nan, np.nan])
        group_n = np.add.reduceat(w, self.starts)
        group_pos = np.add.reduceat(w*self.y, self.starts)
        total_n, total_pos = np.cumsum(group_n), np.cumsum(group_pos)
        precision = np.divide(total_pos, total_n, out=np.zeros_like(total_pos, dtype=float), where=total_n > 0)
        ap = np.sum(group_pos/positives*precision)
        cumulative = np.cumsum(w)
        k = max(1, int(np.ceil(.1*n)))
        boundary = int(np.searchsorted(cumulative, k, side="left"))
        before = cumulative[boundary-1] if boundary else 0
        hits = np.dot(w[:boundary], self.y[:boundary]) + (k-before)*self.y[boundary]
        return np.array([ap, hits/positives])


def bootstrap_differences(raw, y, predictions, reference="hgb_full", B=500):
    weeks = raw._t.dt.to_period("W").astype(str)
    _, codes = np.unique(weeks, return_inverse=True)
    count = codes.max()+1
    tie = raw._priority.to_numpy()
    ranked = {k: RankedMetrics(y, p, tie) for k, p in predictions.items()}
    point = {k: r.weighted(np.ones(len(y), dtype=int)) for k, r in ranked.items()}
    draws = {k: [] for k in ranked if k != reference}
    rng = np.random.default_rng(42)
    for _ in range(B):
        week_weights = rng.multinomial(count, np.full(count, 1/count))
        weights = week_weights[codes]
        ref = ranked[reference].weighted(weights)
        for name in draws:
            draws[name].append(ref-ranked[name].weighted(weights))
    rows = []
    for name, values in draws.items():
        ci = np.nanquantile(np.array(values), [.025, .975], axis=0)
        delta = point[reference]-point[name]
        rows.append({"reference": reference, "competitor": name, "weeks": count, "draws": B,
                     "delta_ap": delta[0], "ap_lower": ci[0,0], "ap_upper": ci[1,0],
                     "delta_recall10": delta[1], "recall10_lower": ci[0,1], "recall10_upper": ci[1,1]})
    return rows


def run(out=ROOT, bootstrap=500):
    out = Path(out)
    result = out / "results"
    result.mkdir(parents=True, exist_ok=True)
    artifacts = artifacts_dir(out)
    artifacts.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw = {s: pd.read_pickle(cache_dir(out) / f"{s}.pkl") for s in ["train", "validation", "test", "stress"]}
    assert all(len(d) for d in raw.values())
    expected = {"train": range(2016,2020), "validation": [2020], "test": [2021], "stress": [2022]}
    for s, d in raw.items():
        assert d.Source.eq("Source2").all() and d._t.dt.year.isin(expected[s]).all()
        assert d._event.is_unique and d.ID.is_unique
    for a, b in itertools.combinations(raw, 2):
        assert not set(raw[a]._event) & set(raw[b]._event), f"Event overlap {a}/{b}"
        assert not set(raw[a].ID) & set(raw[b].ID), f"ID overlap {a}/{b}"
    builder = FeatureBuilder().fit(raw["train"])
    x = {s: builder.transform(d) for s, d in raw.items()}
    y = {s: (d.Severity.to_numpy() >= 3).astype(int) for s, d in raw.items()}
    ties = {s: d._priority.to_numpy() for s, d in raw.items()}
    models, valrows = {}, []
    train, valid = x["train"], x["validation"]
    yt, yv = y["train"], y["validation"]
    full = builder.columns

    def register(name, model, fit_x=train, fit_y=yt, val_x=valid):
        t0 = time.time()
        if not hasattr(model, "fitted") and not hasattr(model, "prior"):
            model.fit(fit_x, fit_y)
        m = score(yv, model.predict(val_x), ties["validation"])
        m.update(name=name, kind=model.kind, columns="|".join(model.columns), features=len(model.columns), seconds=time.time()-t0)
        models[name] = model
        valrows.append(m)
        pd.DataFrame(valrows).to_csv(result / "validation_models.csv", index=False)
        print(format_score(f"Validation {name}", m), flush=True)
        return m

    register("prior", Model("prior", []))
    singles, singlerules = [], {}
    for c in full:
        rule = Model("one", [c]).fit(train, yt)
        m = score(yv, rule.predict(valid), ties["validation"])
        singles.append(dict(variable=c, group=builder.groups[c], **m))
        singlerules[c] = rule
    single_table = pd.DataFrame(singles).sort_values(["ap", "variable"], ascending=[False,True])
    single_table.to_csv(result / "single_variables.csv", index=False)
    best_one = single_table.iloc[0].variable
    register("best_single_rule", singlerules[best_one])
    save_json(result / "all_single_rules.json", {c: m.describe() for c,m in singlerules.items()})

    # Shared training-only encoding. No label encoding of unordered categories.
    pre = preprocessor(train)
    encoded_train = pre.fit_transform(train).tocsc()
    encoded_valid = pre.transform(valid).tocsc()
    cats = [c for c in train if str(train[c].dtype) == "category"]
    nums = [c for c in train if c not in cats]
    column_indices = {c: [i] for i,c in enumerate(nums)}
    offset = len(nums)
    for c, categories in zip(cats, pre.named_transformers_["cat"].categories_):
        column_indices[c] = list(range(offset, offset+len(categories)))
        offset += len(categories)
    pairs = []
    for i, (a, b) in enumerate(itertools.combinations(full, 2)):
        ix = column_indices[a]+column_indices[b]
        tree = DecisionTreeClassifier(max_leaf_nodes=4, max_depth=3, min_samples_leaf=.01, random_state=42)
        tree.fit(encoded_train[:, ix], yt)
        pred = tree.predict_proba(encoded_valid[:, ix])[:,1]
        pairs.append({"a": a, "b": b, **score(yv, pred, ties["validation"])})
        if i % 50 == 0:
            print(f"Pair search: {i+1} tested, elapsed {time.time()-started:.0f}s", flush=True)
    pair_table = pd.DataFrame(pairs).sort_values(["ap", "a", "b"], ascending=[False,True,True])
    pair_table.to_csv(result / "pair_rules.csv", index=False)
    best_pair = pair_table.iloc[0][["a", "b"]].tolist()
    register("best_pair_rule", Model("tree", best_pair))
    del encoded_train, encoded_valid, pre

    tuning = []
    for name, cols in [("logistic_one", [best_one]), ("logistic_pair", best_pair), ("logistic_full", full)]:
        best = None
        for C in [.1, 1., 10.]:
            model = Model("logistic", cols, {"C": C}).fit(train, yt)
            metrics = score(yv, model.predict(valid), ties["validation"])
            ap = metrics["ap"]
            tuning.append({"family": name, "parameters": json.dumps({"C": C}), **metrics})
            if best is None or ap > best[0]:
                best = ap, model
        register(name, best[1])
    best = None
    for leaves, iters in itertools.product([7,15,31], [100,200, 400]):
        params = {"max_leaf_nodes": leaves, "max_iter": iters}
        model = Model("hgb", full, params).fit(train, yt)
        metrics = score(yv, model.predict(valid), ties["validation"])
        ap = metrics["ap"]
        tuning.append({"family": "hgb_full", "parameters": json.dumps(params), **metrics})
        print(format_score(f"HGB {params}", metrics), flush=True)
        if best is None or ap > best[0]:
            best = ap, model
    register("hgb_full", best[1])
    params = best[1].parameters
    tuning_table = pd.DataFrame(tuning).sort_values(["family", "ap"], ascending=[True, False])
    tuning_table.to_csv(result / "tuning.csv", index=False)
    print("\nValidation tuning summary (ranked by AP; no test-period selection):", flush=True)
    print(tuning_table[["family", "parameters", "ap", "auc", "precision10", "recall10", "lift10"]]
          .to_string(index=False, formatters={
              "ap": "{:.4f}".format, "auc": "{:.4f}".format,
              "precision10": "{:.1%}".format, "recall10": "{:.1%}".format,
              "lift10": "{:.2f}x".format}), flush=True)
    nonlinear_pair_rows = []
    for i, row in pair_table.head(5).iterrows():
        name = f"hgb_pair_{len(nonlinear_pair_rows)+1}"
        nonlinear_pair_rows.append(register(name, Model("hgb", [row.a,row.b], params)))
    best_nonlinear_pair = max(nonlinear_pair_rows, key=lambda r:r["ap"])["name"]

    for g in GROUPS:
        groupcols = [c for c in full if builder.groups[c] == g]
        register(f"only_{g}", Model("hgb", groupcols, params))
        register(f"without_{g}", Model("hgb", [c for c in full if c not in groupcols], params))
        added = list(dict.fromkeys(best_pair+groupcols))
        register(f"pair_plus_{g}", Model("hgb", added, params))
    # A fixed validation-selected ordering, not a test-selected feature ranking.
    ordered = best_pair + [c for c in single_table.variable if c not in best_pair]
    for k in [1,2,5,10,len(full)]:
        cols = [best_one] if k == 1 else ordered[:k]
        register(f"budget_{k}", Model("hgb", cols, params))
    nesting = np.random.default_rng(42).permutation(len(train))
    for n in [25000,50000,100000,200000]:
        ix = nesting[:min(n,len(train))]
        register(f"train_size_{n}", Model("hgb", full, params), train.iloc[ix], yt[ix])
    mx = {s: missing_features(frame) for s,frame in x.items()}
    maskcols = [c for c in mx["train"] if mx["train"][c].nunique() > 1]
    register("missingness_only", Model("hgb", maskcols, params), mx["train"], yt, mx["validation"])
    shuffled_y = np.random.default_rng(123).permutation(yt)
    register("shuffled_labels_control", Model("hgb", full, params), train, shuffled_y)
    near_unique_train = ~raw["train"]._near.duplicated().to_numpy()
    register("near_deduplicated_train", Model("hgb", full, params), train.loc[near_unique_train], yt[near_unique_train])

    # Diagnostic-only future information, explicitly outside the admissible competition.
    dx = {}
    for s,d in raw.items():
        dx[s] = pd.DataFrame({"Distance(mi)": pd.to_numeric(d["Distance(mi)"], errors="coerce"),
                             "duration_min": (pd.to_datetime(d.End_Time, format="mixed", errors="coerce")-d._t).dt.total_seconds()/60})
    diagnostic_rows = []
    for c in dx["train"]:
        rule = Model("one", [c]).fit(dx["train"], yt)
        register(f"diagnostic_{c}", rule, dx["train"], yt, dx["validation"])
        # Four-class stump checks whether future information nearly contains Severity itself.
        imp = SimpleImputer(strategy="median").fit(dx["train"][[c]])
        tree = DecisionTreeClassifier(max_leaf_nodes=4, min_samples_leaf=.01, random_state=42).fit(
            imp.transform(dx["train"][[c]]), raw["train"].Severity)
        pred = tree.predict(imp.transform(dx["validation"][[c]]))
        diagnostic_rows.append({"variable": c, "validation_accuracy_4class": float(np.mean(pred==raw["validation"].Severity)),
                                "rule": export_text(tree, feature_names=[c], decimals=4)})
    pd.DataFrame(diagnostic_rows).to_csv(result / "future_information_4class.csv", index=False)
    for s in raw:
        both = pd.concat([x[s],dx[s]],axis=1)
        dx[s] = both
    register("diagnostic_full_future", Model("hgb", full+["Distance(mi)", "duration_min"], params), dx["train"], yt, dx["validation"])

    admissible = [r for r in valrows if r["name"] in ["best_single_rule", "best_pair_rule", "logistic_one", "logistic_pair", best_nonlinear_pair]]
    simple_champion = max(admissible, key=lambda r:r["ap"])["name"]
    frozen = {"selection_metric": "Average Precision on 2020 validation only", "best_single": best_one,
              "best_pair": best_pair, "best_nonlinear_pair": best_nonlinear_pair, "simple_champion": simple_champion,
              "hgb_parameters": params, "feature_order": ordered,
              "feature_groups": builder.groups, "dropped_train_constants": builder.dropped,
              "models": {name:m.describe() for name,m in models.items()},
              "decision_rule": "delta recall10 >= 0.05, paired weekly CI lower > 0, and no AP decrease",
              "test_used_for_selection": False, "bootstrap_draws": bootstrap,
              "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    # This artifact is written before any held-out model scores are computed.
    save_json(result / "frozen_selection.json", frozen)
    joblib.dump({"builder": builder, "models": models}, artifacts / "frozen_models.joblib", compress=3)

    evaluation = []
    predictions = {}
    for s in ["test", "stress"]:
        predframe = pd.DataFrame({"ID": raw[s].ID, "event_key": raw[s]._event, "y": y[s],
                                  "time": raw[s]._t, "State": raw[s].State, "tie": ties[s]})
        for name,model in models.items():
            use = mx[s] if name == "missingness_only" else dx[s] if name.startswith("diagnostic_") else x[s]
            p = model.predict(use)
            evaluation.append({"split":s, "name":name, "features":len(model.columns), **score(y[s],p,ties[s])})
            predframe[name] = p
        predictions[s] = predframe
        predframe.to_csv(artifacts / f"predictions_{s}.csv",index=False)
    pd.DataFrame(evaluation).to_csv(result / "heldout_models.csv",index=False)

    ci_rows = []
    compare = list(dict.fromkeys(["best_single_rule", "best_pair_rule", "logistic_full", best_nonlinear_pair, simple_champion]))
    for s in ["test", "stress"]:
        selected = {name:predictions[s][name].to_numpy() for name in ["hgb_full"]+compare}
        print(f"Paired weekly bootstrap on {s}: {bootstrap} draws", flush=True)
        for r in bootstrap_differences(raw[s],y[s],selected,B=bootstrap):
            ci_rows.append({"split":s, **r})
    pd.DataFrame(ci_rows).to_csv(result / "paired_weekly_intervals.csv",index=False)

    # Permute each whole group together; geographic variables stay internally consistent.
    permutations = []
    for g in GROUPS:
        cols = [c for c in full if builder.groups[c]==g]
        reference_ap = average_precision_score(y["validation"], models["hgb_full"].predict(valid))
        for repeat in range(5):
            z = valid.copy()
            order = np.random.default_rng(100+repeat).permutation(len(z))
            z[cols] = valid.iloc[order][cols].set_axis(z.index)
            ap = average_precision_score(yv,models["hgb_full"].predict(z))
            permutations.append({"group":g,"repeat":repeat,"validation_ap_loss":reference_ap-ap})
    pd.DataFrame(permutations).to_csv(result / "group_permutations.csv",index=False)

    sliced = []
    sensitivity = []
    seen_near = set(raw["train"]._near)
    for s in ["test", "stress"]:
        d = raw[s]
        for dimension, labels in [("month", d._t.dt.strftime("%Y-%m")), ("State", d.State)]:
            for value in labels.unique():
                ix = np.flatnonzero((labels==value).to_numpy())
                if len(ix)<100:
                    continue
                for name in ["hgb_full",simple_champion]:
                    sliced.append({"split":s,"dimension":dimension,"value":value,"name":name,
                                   **score(y[s][ix],predictions[s][name].to_numpy()[ix],ties[s][ix])})
        seen_train = d._near.isin(seen_near).to_numpy()
        duplicate_within = d._near.duplicated().to_numpy()
        keep = ~seen_train & ~duplicate_within
        for name in ["hgb_full",simple_champion,"near_deduplicated_train"]:
            sensitivity.append({"split":s,"name":name,"excluded_near_train":int(seen_train.sum()),
                                "excluded_within_holdout":int(duplicate_within.sum()),
                                **score(y[s][keep],predictions[s][name].to_numpy()[keep],ties[s][keep])})
    pd.DataFrame(sliced).to_csv(result / "monthly_state_scores.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(result / "near_duplicate_sensitivity.csv",index=False)
    # Compare random and temporal splits using development records only. Test stays outside.
    development = pd.concat([raw["train"],raw["validation"]],ignore_index=True)
    order = np.random.default_rng(42).permutation(len(development))
    a,b = order[:len(train)],order[len(train):]
    random_builder = FeatureBuilder().fit(development.iloc[a])
    rx = random_builder.transform(development.iloc[a])
    rv = random_builder.transform(development.iloc[b])
    ry = (development.Severity.to_numpy()>=3).astype(int)
    rm = Model("hgb",random_builder.columns,params).fit(rx,ry[a])
    pd.DataFrame([{"split":"random_development_only",**score(ry[b],rm.predict(rv),development._priority.to_numpy()[b])},
                  {"split":"temporal_2020",**score(yv,models["hgb_full"].predict(valid),ties["validation"])}]).to_csv(result / "random_vs_temporal_development.csv",index=False)
    save_json(result / "verification.json", {"disjoint_ids":True,"disjoint_events":True,"source2_only":True,
        "train_fitted_categories":True,"test_selection":False,"weather_window_minutes":[0,120],
        "test_metrics_computed_after_freeze":True,"paired_bootstrap_draws":bootstrap,
        "elapsed_seconds":time.time()-started})
    print(f"Completed in {time.time()-started:.0f}s. Simple champion: {simple_champion}",flush=True)
    return pd.DataFrame(evaluation)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=ROOT)
    p.add_argument("--bootstrap",type=int,default=500)
    a = p.parse_args()
    with threadpool_limits(limits=4):
        run(a.out,a.bootstrap)
