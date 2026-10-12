"""Bundle the executed final notebook and its dependencies, without raw data."""
from pathlib import Path
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def package():
    notebook_path = ROOT / "notebooks/FINAL_NOTEBOOK.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code), "Execute the notebook first"
    assert not any(o.output_type == "error" for c in code for o in c.outputs)
    assert notebook_path.stat().st_size < 10_000_000
    files = [notebook_path,
        ROOT / "requirements.txt", ROOT / "requirements-main-notebook.txt",
        ROOT / "requirements-final-notebook.txt",
        ROOT / "results/final_notebook_comparison/validation.csv",
        ROOT / "results/final_notebook_comparison/frozen_selection.json"]
    files.extend(sorted((ROOT / "traffic_impact").glob("*.py")))
    files.append(ROOT / "docs/MODE_EMPLOI_NOTEBOOK_FINAL.md")
    files.extend(ROOT / "scripts" / name for name in
        ["build_final_notebook.py", "validate_final_notebook.py", "package_final_notebook.py"])
    output = ROOT / "dist/US_Accidents_Notebook_Final.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert not any(name.startswith(("data/", "artifacts/")) for name in archive.namelist())
    print(f"Created {output.name}: {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    package()
