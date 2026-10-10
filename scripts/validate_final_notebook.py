"""Validate the final notebook and optionally execute it in a fresh local kernel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/FINAL_NOTEBOOK.ipynb"


def validate(execute=False):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        assert chr(0x2014) not in cell.source, index
        if cell.cell_type == "code":
            compile(cell.source, f"final_cell_{index}", "exec")
            assert index > 0 and notebook.cells[index - 1].cell_type == "markdown"
            assert notebook.cells[index + 1].cell_type == "markdown"
    if execute:
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        from nbclient import NotebookClient

        kernel_root = ROOT / "data/final_jupyter/kernels"
        kernel_dir = kernel_root / "final-project"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Final project", "language": "python"}
        (kernel_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
        runtime = ROOT / "data/final_jupyter/runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
        manager = KernelManager(kernel_name="final-project",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))

        def progress(cell, cell_index, **kwargs):
            if cell.cell_type == "code":
                print(f"Executing cell {cell_index}", flush=True)

        client = NotebookClient(notebook, km=manager, timeout=1800, allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress)
        client.execute()
        assert hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() == before, "Notebook edited during execution"
        nbformat.validate(notebook)
        nbformat.write(notebook, NOTEBOOK)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    assert not [o for c in code for o in c.outputs if o.output_type == "error"]
    assert NOTEBOOK.stat().st_size < 10_000_000
    if execute:
        assert all(c.execution_count is not None for c in code)
        import pandas as pd
        checks = pd.read_csv(ROOT / "data/final_notebook/reports/consistency_checks.csv")
        assert checks.Passed.all()
    result = {"notebook": NOTEBOOK.name, "schema_valid": True, "code_cells": len(code),
        "all_cells_executed": all(c.execution_count is not None for c in code),
        "error_outputs": 0, "bytes": NOTEBOOK.stat().st_size}
    folder = ROOT / "data/final_notebook/reports"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "notebook_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    validate(parser.parse_args().execute)
