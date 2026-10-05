"""Additional checks triggered by the nonzero shuffled-label control.

No production model selection is changed. The alternative recall competitor
is chosen using validation scores only; it is reported as a sensitivity.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment import Model, FeatureBuilder, threadpool_limits, bootstrap_differences, score
from .paths import ROOT, artifacts_dir, cache_dir
from .prepare import save_json


def run(out=ROOT):
    out=Path(out); r=out/"results"
    frozen=json.loads((r/"frozen_selection.json").read_text(encoding="utf-8"))
    raw={s:pd.read_pickle(cache_dir(out)/f"{s}.pkl") for s in ["train","test","stress"]}
    builder=FeatureBuilder().fit(raw["train"])
    x={s:builder.transform(d) for s,d in raw.items()}
    y={s:(d.Severity.to_numpy()>=3).astype(int) for s,d in raw.items()}
    val=pd.read_csv(r/"validation_models.csv")
    eligible=val[(val.features<=2)&~val.name.str.startswith("diagnostic_")&~val.name.eq("prior")]
    recall_champion=eligible.sort_values(["recall10","name"],ascending=[False,True]).iloc[0]["name"]
    intervals=[]
    for s in ["test","stress"]:
        pred=pd.read_csv(artifacts_dir(out)/f"predictions_{s}.csv")
        assert pred.ID.tolist()==raw[s].ID.tolist()
        names=list(dict.fromkeys(["hgb_full",recall_champion,"without_weather","without_light","budget_5","budget_10"]))
        for row in bootstrap_differences(raw[s],y[s],{n:pred[n].to_numpy() for n in names},B=500):
            intervals.append({"split":s,**row})
    pd.DataFrame(intervals).to_csv(r/"sensitivity_intervals.csv",index=False)
    rows=[]
    for seed in [321,456,789,999]:
        shuffled=np.random.default_rng(seed).permutation(y["train"])
        model=Model("hgb",builder.columns,frozen["hgb_parameters"]).fit(x["train"],shuffled)
        for s in ["test","stress"]:
            rows.append({"seed":seed,"split":s,**score(y[s],model.predict(x[s]),raw[s]._priority.to_numpy())})
        print(f"Additional null-label control completed, seed={seed}",flush=True)
    baseline=pd.read_csv(r/"heldout_models.csv")
    for s in ["test","stress"]:
        original=baseline[(baseline.split==s)&(baseline.name=="shuffled_labels_control")].iloc[0]
        rows.append({"seed":123,"split":s,**original.drop(labels=["split","name","features"]).to_dict()})
    pd.DataFrame(rows).to_csv(r/"shuffled_label_replicates.csv",index=False)
    save_json(r/"additional_checks.json",{"recall_selected_simple_competitor":recall_champion,
        "selection_data":"2020 validation only; sensitivity, not replacement of predeclared AP selection",
        "extra_shuffle_seeds":[321,456,789,999],"predeclared_ap_selection_unchanged":True})


if __name__=="__main__":
    with threadpool_limits(limits=4):
        run()
