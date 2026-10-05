"""Check artifact consistency and replay saved predictions without refitting."""
import hashlib
import json
from pathlib import Path
import py_compile
import re
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from traffic_impact.paths import artifacts_dir, audit_dir, cache_dir, docs_dir, notebook_path
from traffic_impact.prepare import save_json


def validate(out=ROOT):
    out=Path(out); r=out/"results"
    code=[p for folder in ["traffic_impact","scripts","tests"] for p in sorted((out/folder).glob("*.py"))]
    for path in code:
        py_compile.compile(str(path),doraise=True,cfile=str(Path(tempfile.gettempdir())/"validate_artifacts.pyc"))
        assert chr(0x2014) not in path.read_text(encoding="utf-8"),path.name
    for path in [out/"README.md",docs_dir(out)/"RAPPORT.md",docs_dir(out)/"CELLULES_NOTEBOOK.md",docs_dir(out)/"DOCUMENTATION_CIBLE.md"]:
        assert chr(0x2014) not in path.read_text(encoding="utf-8"),path.name
    definition=(docs_dir(out)/"DOCUMENTATION_CIBLE.md").read_text(encoding="utf-8")
    assert "https://www.kaggle.com/sobhanmoosavi/us-accidents/metadata" in definition
    assert "Severity >= 3" in definition and "3 ou 4" in definition and "1 ou 2" in definition
    assert "frontière opérationnelle" in definition and "98,77 %" in definition
    proposed=(docs_dir(out)/"CELLULES_NOTEBOOK.md").read_text(encoding="utf-8")
    code_cells=re.findall(r"```python\n(.*?)\n```",proposed,re.S)
    assert code_cells
    for i,source in enumerate(code_cells):
        compile(source,f"proposed_notebook_cell_{i}","exec")
    severity_counts=pd.read_csv(audit_dir(out)/"monthly_source_severity.csv")
    assert severity_counts.groupby("Severity")["count"].sum().to_dict()=={1:67366,2:6156981,3:1299337,4:204710}
    scope=severity_counts[(severity_counts.Source=="Source2")&(severity_counts._month>="2016-01")&(severity_counts._month<"2022-01")]
    assert scope.groupby("Severity")["count"].sum().to_dict()=={1:1258,2:2038974,3:1032841,4:12889}
    notebook=json.loads(notebook_path(out).read_text(encoding="utf-8"))
    assert notebook["nbformat"]==4
    assert len({c["id"] for c in notebook["cells"]})==len(notebook["cells"])
    for i,c in enumerate(notebook["cells"]):
        text="".join(c["source"])
        assert chr(0x2014) not in text
        if c["cell_type"]=="code": compile(text,f"notebook_cell_{i}","exec")
    size=notebook_path(out).stat().st_size
    assert size<10000000
    audit=json.loads((audit_dir(out)/"audit.json").read_text(encoding="utf-8"))
    frozen=json.loads((r/"frozen_selection.json").read_text(encoding="utf-8"))
    held=pd.read_csv(r/"heldout_models.csv")
    val=pd.read_csv(r/"validation_models.csv")
    assert not frozen["test_used_for_selection"]
    assert len(pd.read_csv(r/"pair_rules.csv"))==len(frozen["feature_order"])*(len(frozen["feature_order"])-1)//2
    champion=frozen["simple_champion"]
    names=["best_single_rule","best_pair_rule","logistic_one","logistic_pair",frozen["best_nonlinear_pair"]]
    assert val[val.name.isin(names)].sort_values("ap",ascending=False).iloc[0]["name"]==champion
    assert set(held.split)=={"test","stress"}
    assert held.groupby("split").n.nunique().eq(1).all()
    assert held.groupby("split").prevalence.nunique().eq(1).all()
    ci=pd.read_csv(r/"paired_weekly_intervals.csv")
    assert ci.draws.eq(500).all()
    replay=False
    if (artifacts_dir(out)/"frozen_models.joblib").exists() and (cache_dir(out)/"test.pkl").exists():
        import joblib
        from traffic_impact.experiment import Model, FeatureBuilder, OneRule
        artifact=joblib.load(artifacts_dir(out)/"frozen_models.joblib")
        raw=pd.read_pickle(cache_dir(out)/"test.pkl").iloc[:1000]
        x=artifact["builder"].transform(raw)
        pred=pd.read_csv(artifacts_dir(out)/"predictions_test.csv",nrows=1000)
        assert pred.ID.tolist()==raw.ID.tolist()
        for name in ["hgb_full","best_single_rule","best_pair_rule","logistic_full"]:
            np.testing.assert_allclose(artifact["models"][name].predict(x),pred[name],rtol=1e-10,atol=1e-12)
        replay=True
    manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    save_json(r/"artifact_validation.json",{"code_compiles":True,"notebook_compiles":True,"notebook_bytes":size,
        "pairs":len(pd.read_csv(r/"pair_rules.csv")),"selection_replayed_from_validation":True,
        "saved_prediction_replay":replay,"source_sha256":manifest,"full_csv_rows":audit["rows"]})
    include=[out/"README.md",out/"requirements.txt"]+code+sorted((out/"docs").glob("*.md"))+[notebook_path(out)]
    include += sorted(audit_dir(out).glob("*"))
    include += sorted((out/"figures").glob("*.png"))
    include += [p for p in sorted(r.iterdir()) if p.is_file() and p.suffix in [".csv",".json",".txt"]]
    (out/"dist").mkdir(exist_ok=True)
    archive=out/"dist"/"rendu_sans_donnees.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in include:
            z.write(path,Path("traffic_impact_study")/path.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any("cache/" in n or "artifacts/" in n or "predictions_" in n or ".joblib" in n or "US_Accidents_March23.csv" in n for n in z.namelist())
    print(f"OK: {size:,}-byte notebook, {len(include)} packaged files, prediction replay={replay}.")
    return r/"artifact_validation.json"


if __name__=="__main__":validate()
