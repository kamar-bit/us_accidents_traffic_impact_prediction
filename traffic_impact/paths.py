"""Chemins du depot, definis une seule fois. `out` designe toujours la racine du depot."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "US_Accidents_March23.csv"


def cache_dir(out=ROOT):
    """Echantillons nettoyes (train, validation, test, stress), regenerables, hors Git."""
    return Path(out) / "data" / "cache"


def audit_dir(out=ROOT):
    """Audit du CSV complet (petits fichiers, versionnes)."""
    return Path(out) / "results" / "audit"


def results_dir(out=ROOT):
    """Tableaux de resultats (petits fichiers, versionnes)."""
    return Path(out) / "results"


def artifacts_dir(out=ROOT):
    """Predictions individuelles et modeles sauvegardes, volumineux, regenerables, hors Git."""
    return Path(out) / "artifacts"


def docs_dir(out=ROOT):
    return Path(out) / "docs"


def figures_dir(out=ROOT):
    return Path(out) / "figures"


def notebook_path(out=ROOT):
    return Path(out) / "notebooks" / "traffic_impact_study.ipynb"
