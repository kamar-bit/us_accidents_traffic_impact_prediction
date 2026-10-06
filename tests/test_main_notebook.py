"""Boundary tests of the notebook's actual metric and change-ledger functions."""
import ast
import json
import math
from pathlib import Path
import re
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/MAIN_NoteBook.ipynb"


def notebook_functions():
    namespace = {"np": np, "pd": pd, "math": math,
                 "average_precision_score": average_precision_score,
                 "roc_auc_score": roc_auc_score}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    wanted = {"evaluate", "audit_transition", "category_status"}
    found = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook_function", "exec"), namespace)
                found.add(node.name)
    assert found == wanted
    return namespace


class MainNotebookChecks(unittest.TestCase):
    def setUp(self):
        self.functions = notebook_functions()

    def test_constant_scores_match_prevalence_with_reproducible_ties(self):
        target = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        priority = np.arange(len(target))[::-1]
        result = self.functions["evaluate"](target, np.full(len(target), 0.4), priority)
        self.assertAlmostEqual(result["ap"], 2 / 11)
        self.assertEqual(result["auc"], 0.5)
        self.assertEqual(result["selected_count"], 2)
        self.assertEqual(result["precision10"], 0)
        self.assertEqual(result["recall10"], 0)

    def test_perfect_ranking_and_ceiling_budget(self):
        target = np.array([1, 1] + [0] * 9)
        result = self.functions["evaluate"](target, np.linspace(1, 0, 11), np.arange(11))
        for metric in ["ap", "auc", "precision10", "recall10"]:
            self.assertEqual(result[metric], 1)

    def test_one_class_does_not_invent_auc_or_recall(self):
        result = self.functions["evaluate"](np.zeros(10), np.zeros(10), np.arange(10))
        self.assertTrue(np.isnan(result["ap"]))
        self.assertTrue(np.isnan(result["auc"]))
        self.assertTrue(np.isnan(result["recall10"]))
        self.assertEqual(result["precision10"], 0)

    def test_invalid_scores_are_rejected(self):
        for scores in [np.array([np.nan, 0.5]), np.array([-0.1, 0.5]), np.array([1.1, 0.5])]:
            with self.assertRaises(AssertionError):
                self.functions["evaluate"](np.array([0, 1]), scores, np.arange(2))

    def test_change_ledger_counts_mappings_and_unique_events(self):
        namespace = self.functions
        namespace.update(change_rows=[], mapping_rows=[], frequency_rows=[])
        affected = np.zeros(3, dtype=bool)
        before = pd.Series([" North ", None, "NW"], dtype="string")
        stripped = before.str.strip()
        after = stripped.replace({"North": "N"})
        namespace["audit_transition"]("test", "wind", "strip", before, stripped, affected)
        namespace["audit_transition"]("test", "wind", "alias", stripped, after, affected)
        self.assertEqual(sum(row["changed_cells"] for row in namespace["change_rows"]), 2)
        self.assertEqual(int(affected.sum()), 1)
        self.assertEqual(sum(row["count"] for row in namespace["mapping_rows"]), 2)
        self.assertEqual(sum(row["count"] for row in namespace["frequency_rows"]), 12)
        self.assertEqual(after.iloc[2], "NW")
        self.assertTrue(pd.isna(after.iloc[1]))

    def test_missing_assignment_is_reported_as_a_change(self):
        namespace = self.functions
        namespace.update(change_rows=[], mapping_rows=[], frequency_rows=[])
        before = pd.Series([None, "Clear"], dtype="string")
        affected = np.zeros(2, dtype=bool)
        namespace["audit_transition"]("test", "weather", "Missing", before, before.fillna("Missing"), affected)
        self.assertEqual(namespace["mapping_rows"][0]["before"], "<MISSING>")
        self.assertEqual(namespace["mapping_rows"][0]["after"], "Missing")
        self.assertEqual(int(affected.sum()), 1)

    def test_known_rare_unseen_and_missing_are_distinct(self):
        namespace = self.functions
        namespace["seen_in_train"] = {"weather": {"Clear", "Smoke"}}
        namespace["builder"] = type("Builder", (), {"vocab": {"weather": ["Clear"]}})()
        values = pd.Series(["Clear", "Smoke", "NewWeather", None], dtype="string")
        self.assertEqual(namespace["category_status"](values, "weather").tolist(),
                         ["retained", "known_grouped", "unseen", "missing"])

    def test_documented_target_counts_and_percentages_match_full_audit(self):
        root = NOTEBOOK.parent.parent
        text = (root / "docs/DOCUMENTATION_CIBLE.md").read_text(encoding="utf-8")
        counts = pd.read_csv(root / "results/audit/monthly_source_severity.csv")
        entire = counts.groupby("Severity")["count"].sum()
        scope = counts[(counts.Source == "Source2") & counts._month.between("2016-01", "2021-12")]
        scoped = scope.groupby("Severity")["count"].sum()
        rows = re.findall(r"^\| ([1-4]) \| ([\d ]+) \| ([\d,]+) % \| ([\d ]+) \| ([\d,]+) % \|", text, re.M)
        self.assertEqual(len(rows), 4)
        for level, full_count, full_pct, scoped_count, scoped_pct in rows:
            level = int(level)
            self.assertEqual(int(full_count.replace(" ", "")), entire[level])
            self.assertEqual(int(scoped_count.replace(" ", "")), scoped[level])
            self.assertAlmostEqual(float(full_pct.replace(",", ".")), round(100 * entire[level] / entire.sum(), 2))
            self.assertAlmostEqual(float(scoped_pct.replace(",", ".")), round(100 * scoped[level] / scoped.sum(), 2))
        for value in [100 * scoped.loc[[3, 4]].sum() / scoped.sum(),
                      100 * entire.loc[[3, 4]].sum() / entire.sum(),
                      100 * scoped[3] / scoped.loc[[3, 4]].sum()]:
            self.assertIn(f"{value:.2f}".replace(".", ",") + " %", text)

    def test_readme_source_year_rates_match_audit(self):
        root = NOTEBOOK.parent.parent
        text = (root / "README.md").read_text(encoding="utf-8")
        counts = pd.read_csv(root / "results/audit/monthly_source_severity.csv")
        counts["year"] = counts._month.str[:4]
        rows = re.findall(r"^\| (20\d\d) \| ([\d,]+) % \| ([\d,]+) % \|", text, re.M)
        self.assertEqual(len(rows), 7)
        for year, source1, source2 in rows:
            for source, percentage in [("Source1", source1), ("Source2", source2)]:
                frame = counts[counts.year.eq(year) & counts.Source.eq(source)]
                actual = 100 * frame.loc[frame.Severity.ge(3), "count"].sum() / frame["count"].sum()
                self.assertAlmostEqual(float(percentage.replace(",", ".")), round(actual, 2))

    def test_readme_main_model_metrics_match_historical_results(self):
        root = NOTEBOOK.parent.parent
        text = (root / "README.md").read_text(encoding="utf-8")
        rows = pd.read_csv(root / "results/heldout_models.csv")
        main = rows[rows.split.eq("test") & rows.name.eq("hgb_full")].iloc[0]
        match = re.search(r"^\| Gradient boosting complet[^\n]+", text, re.M)
        self.assertIsNotNone(match)
        cells = [item.strip() for item in match.group().split("|")[2:-1]]
        expected = [f"{main.ap:.4f}", f"{main.auc:.4f}",
                    f"{100 * main.recall10:.2f} %", f"{100 * main.precision10:.2f} %"]
        self.assertEqual(cells, [item.replace(".", ",") for item in expected])


if __name__ == "__main__":
    unittest.main(verbosity=2)
