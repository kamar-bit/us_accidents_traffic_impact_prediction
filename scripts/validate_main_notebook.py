"""Validate and optionally execute MAIN_NoteBook in a real, local Jupyter kernel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MAIN_NoteBook.ipynb"


def validate(execute=False):
    import nbformat

    source_before = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        assert chr(0x2014) not in cell.source, index
        if cell.cell_type == "code":
            compile(cell.source, f"main_cell_{index}", "exec")
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    links = re.findall(r"\]\((\.\./[^)#]+)(?:#[^)]*)?\)",
                       "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown"))
    for relative in links:
        assert (NOTEBOOK.parent / relative).resolve().exists(), relative
    started = time.perf_counter()
    if execute:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager

        kernel_root = ROOT / "data/main_jupyter/kernels"
        kernel_dir = kernel_root / "main-project"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "MAIN project validation", "language": "python"}
        (kernel_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
        os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / "data/main_jupyter/runtime")
        Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
        manager = KernelManager(kernel_name="main-project",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))

        def progress(cell, cell_index, **kwargs):
            if cell.cell_type == "code":
                print(f"Executing cell {cell_index}: {cell.source.splitlines()[0][:90]}", flush=True)

        client = NotebookClient(notebook, km=manager, timeout=1800, allow_errors=False,
                                resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress)
        client.execute()
        # A user edit during execution takes precedence over writing generated outputs.
        assert hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() == source_before, "Notebook changed during execution"
        nbformat.validate(notebook)
        nbformat.write(notebook, NOTEBOOK)

    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    executed = all(c.execution_count is not None for c in code_cells)
    errors = [o for c in code_cells for o in c.outputs if o.output_type == "error"]
    assert not errors
    assert NOTEBOOK.stat().st_size < 10_000_000
    if execute:
        assert executed
        import pandas as pd
        checks = pd.read_csv(ROOT / "data/main_notebook/reports/consistency_checks.csv")
        assert checks.passed.all()
    result = {"notebook": NOTEBOOK.name, "schema_valid": True,
              "code_cells": len(code_cells), "all_cells_executed": executed,
              "error_outputs": len(errors), "bytes": NOTEBOOK.stat().st_size,
              "local_links_valid": len(links), "duration_seconds": time.perf_counter() - started}
    reports = ROOT / "data/main_notebook/reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "notebook_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    validate(parser.parse_args().execute)
