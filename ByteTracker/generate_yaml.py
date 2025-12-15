import yaml
from pathlib import Path

# Get the directory of this script
current_script_dir = Path(__file__).resolve().parent

# FIX: go to project root, then ByteTracker_data/output
project_root = current_script_dir.parent  # remove this if needed
dataset_base = project_root / "ByteTracker_data/output"

# Dataset directories
dataset_dir = dataset_base / "train"

# YAML content
data = {
    "train": str(dataset_dir / "images"),
    "val": str(dataset_dir / "images"),
    "nc": 1,
    "names": ["face"]
}

# Output YAML path
yaml_path = dataset_base / "data.yaml"

# Ensure folder exists
dataset_base.mkdir(parents=True, exist_ok=True)

with open(yaml_path, "w") as f:
    yaml.dump(data, f)

print(f"✔ data.yaml created at: {yaml_path}")
