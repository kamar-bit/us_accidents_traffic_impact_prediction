"""Sensitivity: nonlinear geographic pairs missed by shallow-tree screening.

Motivated by geography-only validation performance, not by tuning on test.
This supplementary analysis does not replace the predeclared selection.
"""
import itertools
import json
import pandas as pd
from .experiment import FeatureBuilder, Model, threadpool_limits, score, bootstrap_differences
from .paths import ROOT, artifacts_dir, cache_dir
from .prepare import save_json


def run(out=ROOT):
    out=out
    r=out/"results"
    frozen=json.loads((r/"frozen_selection.json").read_text(encoding="utf-8"))
    raw={s:pd.read_pickle(cache_dir(out)/f"{s}.pkl") for s in ["train","validation","test","stress"]}
    builder=FeatureBuilder().fit(raw["train"])
    x={s:builder.transform(d) for s,d in raw.items()}
    y={s:(d.Severity.to_numpy()>=3).astype(int) for s,d in raw.items()}
    models={}; validation=[]
    for pair in itertools.combinations(["Start_Lat","Start_Lng","State"],2):
        name="geo_pair_"+"_".join(pair)
        model=Model("hgb",pair,frozen["hgb_parameters"]).fit(x["train"],y["train"])
        models[name]=model
        validation.append({"name":name,"columns":"|".join(pair),**score(y["validation"],model.predict(x["validation"]),raw["validation"]._priority.to_numpy())})
        print(f"Geographic pair {pair}: validation AP={validation[-1]['ap']:.4f}",flush=True)
    winner=max(validation,key=lambda row:row["ap"])["name"]
    pd.DataFrame(validation).to_csv(r/"geographic_pairs_validation.csv",index=False)
    save_json(r/"geographic_pairs_frozen.json",{"selected":winner,"selection_data":"2020 only",
        "status":"supplementary sensitivity, not replacement of predeclared comparison"})
    held=[]; intervals=[]
    for s in ["test","stress"]:
        pred=pd.read_csv(artifacts_dir(out)/f"predictions_{s}.csv")
        assert pred.ID.tolist()==raw[s].ID.tolist()
        selected={"hgb_full":pred.hgb_full.to_numpy()}
        for name,model in models.items():
            p=model.predict(x[s])
            held.append({"split":s,"name":name,**score(y[s],p,raw[s]._priority.to_numpy())})
            if name==winner: selected[name]=p
        for row in bootstrap_differences(raw[s],y[s],selected,B=500):
            intervals.append({"split":s,**row})
    pd.DataFrame(held).to_csv(r/"geographic_pairs_heldout.csv",index=False)
    pd.DataFrame(intervals).to_csv(r/"geographic_pairs_intervals.csv",index=False)


if __name__=="__main__":
    with threadpool_limits(limits=4):run()
