import yaml
from pathlib import Path
from utils import constants



# FIX: go to project root, then ByteTracker_data/output
project_root = constants.prj_dir  # remove this if needed
out_dir = constants.output_dir

# Dataset directories
dataset_dir = out_dir / "train"

# YAML content
data = {
    "train": str(dataset_dir / "images"),
    "val": str(dataset_dir / "images"),
    "nc": 1,
    "names": ["face"]
}

# Output YAML path
yaml_path = out_dir / "data.yaml"

# Ensure folder exists
out_dir.mkdir(parents=True, exist_ok=True)

with open(yaml_path, "w") as f:
    yaml.dump(data, f)

print(f"✔ data.yaml created at: {yaml_path}")
