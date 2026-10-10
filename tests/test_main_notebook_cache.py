"""Exercise the actual notebook cache cell across environment changes."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/MAIN_NoteBook.ipynb"


class MainNotebookCacheChecks(unittest.TestCase):
    def run_cache_cell(self, cache_versions, corrupt_sample=False):
        # Read the live notebook rather than testing a copy of its cache logic.
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "".join(next(cell["source"] for cell in notebook["cells"]
                              if cell.get("id") == "main-005"))
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            csv = work / "input.csv"
            csv.write_text("ID\nA-1\n", encoding="utf-8")
            module = work / "prepare.py"
            module.write_text("# preparation fixture\n", encoding="utf-8")
            splits = ["train", "validation", "test", "stress"]
            audit_path = work / "results/audit/audit.json"
            audit_path.parent.mkdir(parents=True)
            cache = work / "data/cache"
            cache.mkdir(parents=True)
            audit = {key: 0 for key in ["rows", "columns", "duplicate_ids", "source2_rows",
                                        "source2_duplicate_event_rows", "source2_conflicting_event_keys"]}
            audit["samples"] = {split: {"rows": 1} for split in splits}

            def rebuild(*args, **kwargs):
                audit_path.write_text(json.dumps(audit), encoding="utf-8")
                for split in splits:
                    pd.DataFrame({"ID": ["A-1"]}).to_pickle(cache / f"{split}.pkl")

            rebuild()
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            signature = {"csv": str(csv.resolve()), "bytes": csv.stat().st_size,
                         "mtime_ns": csv.stat().st_mtime_ns,
                         "preparation_sha256": sha(module),
                         "limits": dict(zip(splits, [200000, 50000, 100000, 50000])), "seed": 42}
            if cache_versions is not None:
                signature["cache_versions"] = cache_versions
            manifest = {"signature": signature,
                        "sample_sha256": {split: sha(cache / f"{split}.pkl") for split in splits}}
            (work / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            if corrupt_sample:
                (cache / "train.pkl").write_bytes(b"broken pickle")
            prepare = SimpleNamespace(__file__=str(module), run=Mock(side_effect=rebuild))
            namespace = dict(Path=Path, hashlib=hashlib, json=json, sys=sys, np=np, pd=pd,
                             CSV_PATH=csv, WORK=work, prepare=prepare, display=lambda value: None)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, "notebook_cache_cell", "exec"), namespace)
            self.assertEqual(set(namespace["raw"]), set(splits))
            self.assertTrue(all(frame.ID.tolist() == ["A-1"] for frame in namespace["raw"].values()))
            return prepare.run.call_count

    def current_versions(self):
        return {"python": list(sys.version_info[:2]), "pandas": pd.__version__, "numpy": np.__version__}

    def test_matching_environment_reuses_samples(self):
        self.assertEqual(self.run_cache_cell(self.current_versions()), 0)

    def test_legacy_or_changed_environment_rebuilds_samples(self):
        versions = self.current_versions()
        cases = [None] + [dict(versions, **{library: "different-version"})
                          for library in ["python", "pandas", "numpy"]]
        for saved_versions in cases:
            with self.subTest(saved_versions=saved_versions):
                self.assertEqual(self.run_cache_cell(saved_versions), 1)

    def test_corrupt_sample_is_rebuilt_before_loading(self):
        self.assertEqual(self.run_cache_cell(self.current_versions(), corrupt_sample=True), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
