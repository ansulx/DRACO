"""Shared paths for the OCT-DR project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REGISTRY_PATH = DATA_DIR / "registry.yaml"

MMRDR_DIR = RAW_DIR / "mmrdr"
OEFI_DIR = RAW_DIR / "oefi"

FIGSHARE_MMRDR_ARTICLE = "29423747"
