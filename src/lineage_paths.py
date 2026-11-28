"""Lineage artifact locations.

Deliberately not in paths.py: that module is a dependency of the training
stages, so adding a constant there would invalidate their cache (a ~90 minute
retrain) for a change that cannot affect the model.
"""

from pathlib import Path

# Recipe of every synthetic sample: the signal recording + offset and the noise
# recording + offset it was mixed from. A DVC output of augmentation, kept apart
# from augmented_annotations.csv so recording it cannot perturb the chunks.
LINEAGE_DIR = Path("./data/lineage")
AUGMENTATION_LINEAGE = LINEAGE_DIR / "augmentation_lineage.csv"

# Derived views (src/lineage.py): disposable, git-ignored, not a DVC stage.
LINEAGE_REPORT_DIR = Path("./data/lineage_report")
