"""Unified 3-class DR severity labels for GHOTA (glaucoma excluded).

Classes:
  0 = none
  1 = early_mod   (raw grades 1+2+3)
  2 = advanced    (raw grades 4+5 / CI-DME)
"""

from __future__ import annotations

GHOTA_CLASS_NAMES = ("none", "early_mod", "advanced")
NUM_CLASSES = 3

# Task IDs within DR family (modality / clinical endpoint)
TASKS = (
    "oct_dme",    # OCT diabetic macular edema staging
    "fundus_dr",  # fundus ICDR-style DR grading
)

# Back-compat alias used in older scaffold code
DISEASES = TASKS

_MAPS: dict[str, dict[int, int]] = {
    # MMRDR No / NCI / CI
    "oct_dme": {0: 0, 1: 1, 2: 2},
    "dme": {0: 0, 1: 1, 2: 2},
    # ICDR 0–5 fundus DR
    "fundus_dr": {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2},
    "dr": {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2},
    # OCTID binary Normal/DR → none / early_mod (severity unknown)
    "octid_binary": {0: 0, 1: 1},
    # OEFI binary No-DME / DME → none / early_mod (external, not for hard Acc)
    "oefi_binary": {0: 0, 1: 1},
}


def map_label(raw: int, task: str, *, scheme: str | None = None) -> int:
    """Map a dataset-native label into GHOTA {0,1,2}."""
    key = scheme or task
    # legacy catalog column was "disease"
    if key == "dme":
        key = "oct_dme"
    if key == "dr":
        key = "fundus_dr"
    table = _MAPS.get(key)
    if table is None:
        raise KeyError(
            f"No GHOTA map for task={task!r} scheme={scheme!r}. "
            f"Known: {sorted(_MAPS)}. Glaucoma is handled by GLAUCO, not GHOTA."
        )
    raw = int(raw)
    if raw not in table:
        raise ValueError(f"Raw label {raw} not in map {key}: {sorted(table)}")
    return table[raw]


def class_name(label: int) -> str:
    return GHOTA_CLASS_NAMES[int(label)]
