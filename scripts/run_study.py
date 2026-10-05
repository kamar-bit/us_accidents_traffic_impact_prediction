"""One entry point, cached audit, explicit rerun of the frozen study."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_impact.paths import ROOT, DEFAULT_CSV, artifacts_dir, audit_dir, cache_dir, results_dir  # noqa: E402
from traffic_impact.prepare import run as prepare  # noqa: E402


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--csv",type=Path,default=DEFAULT_CSV)
    p.add_argument("--out",type=Path,default=ROOT)
    p.add_argument("--rerun",action="store_true",help="Refit the predeclared study and recompute held-out scores")
    a=p.parse_args(argv)
    audit_path=audit_dir(a.out)/"audit.json"
    current=False
    if audit_path.exists():
        audit=json.loads(audit_path.read_text(encoding="utf-8"))
        current=(audit.get("file")==str(a.csv.resolve()) and audit.get("file_bytes")==a.csv.stat().st_size
                 and audit.get("file_mtime_ns")==a.csv.stat().st_mtime_ns
                 and all((cache_dir(a.out)/f"{s}.pkl").exists() for s in ["train","validation","test","stress"]))
    should_fit = a.rerun or not current or not (a.out/"results"/"verification.json").exists()
    if not current:
        prepare(a.csv,a.out)
    if should_fit:
        from traffic_impact.experiment import run, threadpool_limits
        with threadpool_limits(limits=4):
            run(a.out)
    # Supplementary checks are kept separate from the predeclared primary result.
    from traffic_impact.experiment import threadpool_limits
    with threadpool_limits(limits=4):
        if should_fit or not (a.out/"results"/"additional_checks.json").exists():
            from traffic_impact.verify_results import run as verify
            verify(a.out)
        if should_fit or not (a.out/"results"/"geographic_pairs_frozen.json").exists():
            from traffic_impact.check_geographic_pairs import run as geographic
            geographic(a.out)
    from traffic_impact.report import render
    return render(a.out)


if __name__=="__main__":
    main()
