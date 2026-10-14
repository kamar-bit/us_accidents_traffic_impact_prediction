"""Checks of the final notebook generator and its data dictionary, without the CSV."""
import ast
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_final_notebook import build
from traffic_impact.experiment import LIGHT, POI, WCAT, WNUM
from traffic_impact.final_notebook import data_dictionary


def category_preparation(notebook):
    for cell in notebook.cells:
        if cell.cell_type == "code" and "class CategoryPreparation" in cell.source:
            namespace = {"pd": pd}
            exec(compile(cell.source.split("category_prep =")[0], "category_preparation", "exec"), namespace)
            return namespace["CategoryPreparation"]
    raise AssertionError("CategoryPreparation not found")


class FinalNotebookChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = build()

    def test_every_code_cell_sits_between_markdown_cells(self):
        cells = self.notebook.cells
        for index, cell in enumerate(cells):
            self.assertNotIn(chr(0x2014), cell.source)
            if cell.cell_type == "code":
                ast.parse(cell.source)
                self.assertEqual(cells[index - 1].cell_type, "markdown")
                self.assertEqual(cells[index + 1].cell_type, "markdown")

    def test_inputs_are_33_without_turning_loop(self):
        text = "\n".join(cell.source for cell in self.notebook.cells)
        self.assertNotIn("34 inputs, target excluded", text)
        self.assertNotIn("34 logical", text)
        self.assertNotIn("34-input", text)
        self.assertIn("33 inputs, target excluded", text)
        self.assertEqual(4 + 4 + len(POI) - 1 + len(LIGHT) + len(WNUM) + len(WCAT), 33)

    def test_category_preparation_drops_constant_turning_loop(self):
        preparation = category_preparation(self.notebook)
        frame = pd.DataFrame({"Turning_Loop": [0.0, 0.0, 0.0], "Junction": [0.0, 1.0, 0.0],
                              "State": pd.array(["CA", "NY", "CA"], dtype="string")})
        prepared = preparation().fit(frame)
        self.assertEqual(prepared.columns, ["Junction", "State"])
        self.assertNotIn("Turning_Loop", prepared.transform(frame))

    def test_logistic_regression_is_compared_with_the_trees(self):
        code = "\n".join(c.source for c in self.notebook.cells if c.cell_type == "code")
        self.assertIn("for C in [0.1, 1, 10]", code)
        self.assertIn('"Logistic": logistic.predict_proba', code)
        self.assertIn('predictions["validation"].update(predict_all("validation"))', code)
        self.assertIn('ranking[["Constant", "Logistic", "Forest", "Boosting"]].plot.bar', code)
        self.assertIn('scoring="average_precision", n_repeats=5', code)

    def test_other_boosting_libraries_are_reported(self):
        code = "\n".join(c.source for c in self.notebook.cells if c.cell_type == "code")
        self.assertIn("results/lightgbm_comparison/scores.csv", code)
        self.assertIn('["Boosting", "LightGBM", "XGBoost"]', code)

    def test_errors_are_reported_by_hour_and_by_state(self):
        code = "\n".join(c.source for c in self.notebook.cells if c.cell_type == "code")
        self.assertIn('error_rates(errors, "Start hour")', code)
        self.assertIn('error_rates(errors[errors.State.isin(top_states)], "State")', code)
        self.assertIn("errors_by_state.csv", code)

    def test_wind_chill_fill_is_tested_once_in_the_forest_section(self):
        text = "\n".join(cell.source for cell in self.notebook.cells)
        self.assertEqual(text.count("def fill_wind_chill"), 1)
        self.assertNotIn("2.12 Learn categorical", text)
        self.assertIn("2.11 Learn categorical vocabularies", text)

    def test_related_work_cites_the_public_projects_that_were_read(self):
        text = "\n".join(cell.source for cell in self.notebook.cells)
        for name in ["Atharva-Penkar", "Sams-Alt-Personality", "rayenchanchah01"]:
            self.assertIn(name, text)

    def test_dictionary_marks_turning_loop_excluded(self):
        dictionary = data_dictionary(["Turning_Loop", "Junction"]).set_index("Column")
        self.assertEqual(dictionary.loc["Turning_Loop", "Role"], "excluded")
        self.assertEqual(dictionary.loc["Junction", "Role"], "input")


if __name__ == "__main__":
    unittest.main()
