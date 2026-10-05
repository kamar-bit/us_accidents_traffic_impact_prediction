"""Full streaming audit and uniform event sampling. No model fitting here."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .paths import ROOT, audit_dir, cache_dir
LIMITS = {"train": 200000, "validation": 50000, "test": 100000, "stress": 50000}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def priorities(keys, seed=42):
    """SplitMix64: deterministic priorities independent of labels and file order."""
    z = keys.astype(np.uint64) + np.uint64(seed)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return z ^ (z >> np.uint64(31))


def event_keys(frame):
    # Description is used for record identity only, never as a predictor.
    fields = ["Source", "Start_Time", "End_Time", "Start_Lat", "Start_Lng",
              "Distance(mi)", "Description"]
    return pd.util.hash_pandas_object(frame[fields], index=False).to_numpy()


def near_keys(frame):
    t = pd.to_datetime(frame["Start_Time"], format="mixed", errors="coerce")
    q = pd.DataFrame({"time_bin": t.astype("int64") // (15 * 60 * 10**9),
                      "lat_bin": np.floor(frame["Start_Lat"] * 1000),
                      "lng_bin": np.floor(frame["Start_Lng"] * 1000)}, index=frame.index)
    return pd.util.hash_pandas_object(q, index=False).to_numpy()


def run(csv, out=ROOT, limits=None):
    limits = limits or LIMITS
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = cache_dir(out)
    cache.mkdir(parents=True, exist_ok=True)
    audit = audit_dir(out)
    audit.mkdir(parents=True, exist_ok=True)
    start = time.time()
    n = 0
    missing = None
    monthly = []
    monthly_missing = []
    identifiers, events, event_labels, near, event_months = [], [], [], [], []
    pools = {k: pd.DataFrame() for k in limits}
    available = {k: 0 for k in limits}
    weather = {"observed": 0, "after_start": 0, "eligible_two_hours": 0}
    dur_neg = 0
    value_counts = {c: pd.Series(dtype="int64") for c in ["Country", "Turning_Loop"]}
    for number, ch in enumerate(pd.read_csv(csv, chunksize=250000, low_memory=False)):
        n += len(ch)
        m = ch.isna().sum()
        missing = m if missing is None else missing.add(m, fill_value=0)
        identifiers.append(pd.util.hash_pandas_object(ch.ID, index=False).to_numpy())
        t = pd.to_datetime(ch.Start_Time, format="mixed", errors="coerce")
        end = pd.to_datetime(ch.End_Time, format="mixed", errors="coerce")
        w = pd.to_datetime(ch.Weather_Timestamp, format="mixed", errors="coerce")
        lag = (t - w).dt.total_seconds() / 60
        weather["observed"] += int(lag.notna().sum())
        weather["after_start"] += int((lag < 0).sum())
        weather["eligible_two_hours"] += int(lag.between(0, 120).sum())
        dur_neg += int((end < t).sum())
        ch["_month"] = t.dt.strftime("%Y-%m")
        monthly.append(ch.groupby(["_month", "Source", "Severity"], dropna=False).size())
        for c in value_counts:
            value_counts[c] = value_counts[c].add(ch[c].value_counts(dropna=False), fill_value=0)
        s = ch[ch.Source == "Source2"].copy()
        if len(s):
            s["_event"] = event_keys(s)
            s["_near"] = near_keys(s)
            s["_priority"] = priorities(s._event.to_numpy())
            s["_t"] = t.loc[s.index]
            events.append(s._event.to_numpy())
            near.append(s._near.to_numpy())
            event_labels.append(s.Severity.to_numpy(dtype=np.int8))
            event_months.append(s._month.to_numpy())
            monthly_missing.append(s.groupby("_month")[missing.index.tolist()].agg(lambda x: x.isna().sum()))
            year = s._t.dt.year
            masks = {"train": year.between(2016, 2019), "validation": year == 2020,
                     "test": year == 2021, "stress": year == 2022}
            for split, mask in masks.items():
                z = s.loc[mask].drop(columns=["Description", "_month"])
                available[split] += len(z)
                if len(z):
                    pool = pd.concat([pools[split], z], ignore_index=True)
                    # Equal event priorities give uniform sampling of distinct events.
                    pool = pool.drop_duplicates("_event", keep="first")
                    pools[split] = pool.nsmallest(limits[split], "_priority")
        if number % 4 == 0:
            print(f"Audit: {n:,} records, {time.time()-start:.0f}s", flush=True)

    all_ids = np.concatenate(identifiers)
    del identifiers
    ids_duplicate = int(len(all_ids) - len(np.unique(all_ids)))
    del all_ids
    keys = np.concatenate(events)
    labels = np.concatenate(event_labels)
    months = np.concatenate(event_months)
    nk = np.concatenate(near)
    order = np.argsort(keys)
    sk, sy = keys[order], labels[order]
    starts = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
    conflicting = sk[starts][np.minimum.reduceat(sy, starts) != np.maximum.reduceat(sy, starts)]
    duplicate_counts = np.diff(np.r_[starts, len(sk)])
    _, first = np.unique(keys, return_index=True)
    duplicate_month = pd.Series(months[np.setdiff1d(np.arange(len(keys)), first)]).value_counts()
    pd.DataFrame({"duplicate_event_rows": duplicate_month}).sort_index().to_csv(audit / "duplicates_by_month.csv")
    monthly_table = pd.concat(monthly).groupby(level=[0, 1, 2], dropna=False).sum().rename("count").reset_index()
    monthly_table.to_csv(audit / "monthly_source_severity.csv", index=False)
    pd.concat(monthly_missing).groupby(level=0).sum().to_csv(audit / "monthly_missing_source2.csv")
    profile = pd.DataFrame({"column": missing.index, "missing": missing.to_numpy().astype(int),
                            "missing_pct": 100 * missing.to_numpy() / n})
    profile.to_csv(audit / "columns.csv", index=False)
    summary = {"file": str(Path(csv).resolve()), "file_bytes": Path(csv).stat().st_size,
               "file_mtime_ns": Path(csv).stat().st_mtime_ns,
               "rows": n, "columns": len(missing), "duplicate_ids": ids_duplicate,
               "source2_rows": len(keys), "source2_duplicate_event_rows": int(duplicate_counts.sum()-len(starts)),
               "source2_conflicting_event_keys": len(conflicting),
               "source2_near_collision_rows": int(len(nk)-len(np.unique(nk))),
               "identity_definition": "Source, Start_Time, End_Time, Start_Lat, Start_Lng, Distance(mi), Description",
               "near_definition": "15-minute bins, 0.001-degree spatial cells (approximate, boundary-sensitive)",
               "hash_note": "64-bit hashes; vanishingly rare collisions are possible",
               "negative_durations": dur_neg, "weather": weather,
               "constant_columns": [c for c, v in value_counts.items() if len(v) == 1],
               "available_before_cleaning": available, "seed": 42, "samples": {}}
    for split, pool in pools.items():
        pool = pool[~pool._event.isin(conflicting)].sort_values("_t").reset_index(drop=True)
        pool.to_pickle(cache / f"{split}.pkl")
        summary["samples"][split] = {"rows": len(pool), "high_impact_prevalence": float((pool.Severity >= 3).mean()),
                                      "first": str(pool._t.min()), "last": str(pool._t.max())}
    save_json(audit / "audit.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, default=ROOT)
    a = p.parse_args()
    run(a.csv, a.out)
