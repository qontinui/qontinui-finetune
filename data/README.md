# Data Organization and Structure

## Overview

This directory contains all datasets and annotations used for training, validating, and testing vision models in the qontinui-finetune project. This README documents the organization, structure, and management of datasets.

---

## Directory Structure

```
data/
├── datasets/                    # Training datasets organized by domain
│   ├── game-ui/                 # [TO BE FILLED: Game UI screenshots]
│   ├── gui-elements/            # [TO BE FILLED: GUI element database]
│   └── web-ui/                  # [TO BE FILLED: Web UI screenshots]
├── annotations/                 # Annotation files and metadata
│   ├── game-ui/
│   ├── gui-elements/
│   ├── web-ui/
│   ├── class_definitions.json
│   └── split_info.json
├── README.md                    # This file
└── [other directories as needed]
```

---

## Datasets

### 1. Game UI Dataset

**Location:** `/home/user/qontinui-finetune/data/datasets/game-ui/`

**Description:** [TO BE FILLED]
- Purpose: [e.g., Training on game interface detection]
- Size: [TO BE FILLED] images
- Source: [TO BE FILLED]
- Created: [YYYY-MM-DD or TO BE FILLED]

**Contents:**
```
game-ui/
├── [image_file_1].jpg
├── [image_file_2].jpg
├── ...
└── metadata.json
```

**Metadata example:**
```json
{
  "dataset_name": "game-ui",
  "total_images": [#],
  "resolution_range": {
    "min": [width, height],
    "max": [width, height],
    "average": [width, height]
  },
  "source": "[Source description]",
  "annotation_type": "bounding_boxes / segmentation_masks",
  "classes": ["button", "menu", "text_field", ...]
}
```

**Usage notes:**
- [TO BE FILLED: Any specific notes about this dataset]

---

### 2. GUI Elements Dataset

**Location:** `/home/user/qontinui-finetune/data/datasets/gui-elements/`

**Description:** [TO BE FILLED]
- Purpose: [e.g., Database of isolated GUI components]
- Size: [TO BE FILLED] images
- Source: [TO BE FILLED]
- Created: [YYYY-MM-DD or TO BE FILLED]

**Contents:**
```
gui-elements/
├── buttons/
│   ├── button_001.jpg
│   ├── button_002.jpg
│   └── ...
├── text_fields/
│   ├── textfield_001.jpg
│   └── ...
├── menus/
│   └── ...
└── metadata.json
```

**Subdirectory organization:**

| Subdirectory | Count | Description |
|--------------|-------|-------------|
| buttons | [#] | [TO BE FILLED] |
| text_fields | [#] | [TO BE FILLED] |
| menus | [#] | [TO BE FILLED] |
| [other] | [#] | [TO BE FILLED] |

**Usage notes:**
- [TO BE FILLED: Any specific notes about this dataset]

---

### 3. Web UI Dataset

**Location:** `/home/user/qontinui-finetune/data/datasets/web-ui/`

**Description:** [TO BE FILLED]
- Purpose: [e.g., Web application interface detection]
- Size: [TO BE FILLED] images
- Source: [TO BE FILLED]
- Created: [YYYY-MM-DD or TO BE FILLED]

**Contents:**
```
web-ui/
├── [website_1]/
│   ├── screenshot_001.png
│   ├── screenshot_002.png
│   └── ...
├── [website_2]/
│   └── ...
└── metadata.json
```

**Usage notes:**
- [TO BE FILLED: Any specific notes about this dataset]

---

## Annotations

### 1. Annotation Files

**Location:** `/home/user/qontinui-finetune/data/annotations/`

**Structure:**
```
annotations/
├── game-ui/
│   ├── image_001.json
│   ├── image_002.json
│   └── ...
├── gui-elements/
│   └── ...
├── web-ui/
│   └── ...
├── class_definitions.json
└── split_info.json
```

### 2. Annotation Format

**Primary format:** [YOLO/COCO/Pascal VOC/Custom] [TO BE FILLED]

**Example annotation file:**

```json
{
  "image_id": "game-ui_001",
  "image_path": "datasets/game-ui/game-ui_001.jpg",
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "annotations": [
    {
      "id": 1,
      "category_id": 1,
      "category_name": "button",
      "bbox": [x, y, width, height],
      "area": 10000,
      "segmentation": [optional_polygon_coordinates],
      "iscrowd": 0,
      "confidence": 1.0
    },
    {
      "id": 2,
      "category_id": 2,
      "category_name": "text_field",
      "bbox": [x, y, width, height],
      "area": 5000,
      "segmentation": null,
      "iscrowd": 0,
      "confidence": 1.0
    }
  ]
}
```

### 3. Class Definitions

**Location:** `/home/user/qontinui-finetune/data/annotations/class_definitions.json`

```json
{
  "classes": {
    "1": {
      "name": "button",
      "description": "Clickable UI button",
      "color": "#FF6B6B",
      "examples": ["TO BE FILLED"]
    },
    "2": {
      "name": "text_field",
      "description": "Text input field",
      "color": "#4ECDC4",
      "examples": ["TO BE FILLED"]
    },
    "3": {
      "name": "menu",
      "description": "Navigation menu",
      "color": "#FFE66D",
      "examples": ["TO BE FILLED"]
    }
  },
  "total_classes": 3,
  "annotation_type": "bounding_boxes",
  "last_updated": "YYYY-MM-DD"
}
```

### 4. Dataset Splits

**Location:** `/home/user/qontinui-finetune/data/annotations/split_info.json`

```json
{
  "split_strategy": "random",
  "random_seed": 42,
  "split_date": "YYYY-MM-DD",
  "splits": {
    "train": {
      "total_images": [#],
      "image_ids": ["image_id_1", "image_id_2", ...],
      "percentage": 70
    },
    "val": {
      "total_images": [#],
      "image_ids": ["image_id_X", "image_id_Y", ...],
      "percentage": 15
    },
    "test": {
      "total_images": [#],
      "image_ids": ["image_id_A", "image_id_B", ...],
      "percentage": 15
    }
  },
  "total_images": [#]
}
```

---

## Data Statistics

### Dataset Summary

| Dataset | Images | Annotations | Classes | Split |
|---------|--------|-------------|---------|-------|
| game-ui | [#] | [#] | [#] | Train/Val/Test |
| gui-elements | [#] | [#] | [#] | [TO BE FILLED] |
| web-ui | [#] | [#] | [#] | [TO BE FILLED] |
| **Total** | [#] | [#] | - | - |

### Class Distribution

**Overall class distribution across all datasets:**

```
button           ████████████████░░ (45%)  [# annotations]
text_field       ██████████░░░░░░░░ (28%)  [# annotations]
menu             ████████░░░░░░░░░░ (18%)  [# annotations]
dialog           ████░░░░░░░░░░░░░░ (9%)   [# annotations]
```

**Per-dataset class distribution:**

| Class | game-ui | gui-elements | web-ui | Total |
|-------|---------|--------------|--------|-------|
| button | [#] | [#] | [#] | [#] |
| text_field | [#] | [#] | [#] | [#] |
| menu | [#] | [#] | [#] | [#] |

### Image Statistics

**Image dimensions:**
```
Minimum resolution:  [width] x [height]
Maximum resolution:  [width] x [height]
Average resolution:  [width] x [height]
Most common size:    [width] x [height] ([count] images)
```

**File sizes:**
```
Total data size:      [GB]
Average image size:   [MB]
Minimum image size:   [KB]
Maximum image size:   [MB]
```

---

## Data Management

### 1. Version Control

**Dataset versioning:**

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| 1.0 | YYYY-MM-DD | Initial dataset | Active |
| 1.1 | YYYY-MM-DD | [Changes] | [TO BE FILLED] |

**Version information:**
- Current version: [#]
- Last updated: YYYY-MM-DD
- Maintainer: [TO BE FILLED]

### 2. Data Access

**File permissions:**
```bash
# View permissions
ls -la /home/user/qontinui-finetune/data/

# Expected permissions
drwxr-xr-x  - datasets/
drwxr-xr-x  - annotations/
```

**Data location on disk:**
- Primary storage: `/home/user/qontinui-finetune/data/`
- Backup location: [TO BE FILLED if applicable]
- Archive location: [TO BE FILLED if applicable]

### 3. Data Cleanup

**Temporary files to remove before sharing:**
```bash
# Files to exclude from version control
*.tmp
*.backup
__pycache__/
.DS_Store
.ipynb_checkpoints/
```

**Disk usage:**
```bash
# Check disk usage
du -sh /home/user/qontinui-finetune/data/

# Expected output
[#]G total
```

---

## Data Quality Assurance

### 1. Validation Checks

**Automated validation:**

```bash
# Validate dataset structure
python [REFERENCE TO: scripts/validate_dataset.py] \
  --data-dir /home/user/qontinui-finetune/data/ \
  --check-images \
  --check-annotations \
  --check-splits
```

**Checks performed:**
- [ ] All images are readable and not corrupted
- [ ] All annotations reference existing images
- [ ] All class IDs are valid
- [ ] Bounding boxes are within image boundaries
- [ ] No duplicate images
- [ ] All required files present
- [ ] Consistent annotation format

### 2. Quality Metrics

**Dataset quality report:**

| Metric | Value | Status |
|--------|-------|--------|
| Corrupted images | [#] | [OK/WARNING] |
| Missing annotations | [#] | [OK/WARNING] |
| Invalid annotations | [#] | [OK/WARNING] |
| Duplicates | [#] | [OK/WARNING] |
| Overall quality | [%] | [OK/NEEDS REVIEW] |

### 3. Issues and Resolution

**Known issues:**

| Issue | Dataset | Count | Severity | Resolution |
|-------|---------|-------|----------|-----------|
| [Issue 1] | [dataset] | [#] | High/Medium/Low | [TO BE FILLED] |

---

## Usage Guide

### 1. Loading Data in Code

**Python example (PyTorch):**

```python
from pathlib import Path
import json
from PIL import Image

# Define paths
data_dir = Path("/home/user/qontinui-finetune/data")
dataset_name = "game-ui"

# Load class definitions
class_defs_path = data_dir / "annotations" / "class_definitions.json"
with open(class_defs_path) as f:
    class_definitions = json.load(f)

# Load image
image_path = data_dir / "datasets" / dataset_name / "image_001.jpg"
image = Image.open(image_path)

# Load annotations
annotation_path = data_dir / "annotations" / dataset_name / "image_001.json"
with open(annotation_path) as f:
    annotations = json.load(f)

# Access data
for annotation in annotations["annotations"]:
    class_name = annotation["category_name"]
    bbox = annotation["bbox"]
    print(f"Found {class_name} at {bbox}")
```

### 2. Data Access Patterns

**Access specific dataset:**
```python
dataset_path = Path("/home/user/qontinui-finetune/data/datasets/game-ui")
images = list(dataset_path.glob("*.jpg"))
```

**Load splits:**
```python
split_info_path = Path("/home/user/qontinui-finetune/data/annotations/split_info.json")
with open(split_info_path) as f:
    splits = json.load(f)

train_ids = splits["splits"]["train"]["image_ids"]
val_ids = splits["splits"]["val"]["image_ids"]
test_ids = splits["splits"]["test"]["image_ids"]
```

### 3. Creating Subsets

**Create training subset:**
```bash
# [TO BE FILLED: Instructions for creating subsets]
python scripts/create_subset.py \
  --source-dir /home/user/qontinui-finetune/data/ \
  --output-dir /home/user/qontinui-finetune/data/subsets/ \
  --sample-size 100 \
  --random-seed 42
```

---

## Data Privacy and Ethics

### 1. Privacy Considerations

- [TO BE FILLED: Privacy notes for UI screenshots]
- [TO BE FILLED: GDPR/CCPA compliance notes if applicable]
- [TO BE FILLED: User consent information]

### 2. Licensing

**Dataset sources and licenses:**

| Dataset | License | Source | Usage Restrictions |
|---------|---------|--------|-------------------|
| game-ui | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| gui-elements | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| web-ui | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |

### 3. Data Usage Rights

- [TO BE FILLED: Rights and permissions]
- [TO BE FILLED: Attribution requirements]
- [TO BE FILLED: Restrictions on distribution]

---

## Maintenance and Updates

### 1. Regular Maintenance Tasks

**Weekly:**
- [ ] Monitor disk usage
- [ ] Check for corrupted files

**Monthly:**
- [ ] Run validation checks
- [ ] Review data quality metrics
- [ ] Check for duplicates

**Quarterly:**
- [ ] Update class distributions
- [ ] Review and update documentation
- [ ] Assess need for additional data

### 2. Archiving Old Data

**Archive procedure:**
```bash
# [TO BE FILLED: Archiving instructions]
# Example:
tar -czf /archive/qontinui_data_v1.0.tar.gz \
  /home/user/qontinui-finetune/data/
```

### 3. Data Requests

**To request additional data or report issues:**
- [TO BE FILLED: Contact information]
- [TO BE FILLED: Issue tracking system]
- [TO BE FILLED: Data request process]

---

## Tools and Scripts

**Available data management scripts:**

| Script | Location | Purpose |
|--------|----------|---------|
| validate_dataset.py | `/home/user/qontinui-finetune/scripts/` | Validate dataset structure and contents |
| create_subset.py | `/home/user/qontinui-finetune/scripts/` | Create dataset subsets |
| analyze_dataset.py | `/home/user/qontinui-finetune/scripts/` | Generate dataset statistics |
| [Other scripts] | [TO BE FILLED] | [Purpose] |

---

## Related Documentation

- `/home/user/qontinui-finetune/docs/dataset-creation.md` - Dataset creation guide
- `/home/user/qontinui-finetune/docs/training-guide.md` - Training workflow
- `/home/user/qontinui-finetune/README.md` - Project overview
- `/home/user/qontinui-finetune/requirements.txt` - Project dependencies

---

## Appendix

### A. Directory Tree

```
data/
├── datasets/
│   ├── game-ui/
│   │   ├── game-ui_001.jpg
│   │   ├── game-ui_002.jpg
│   │   └── metadata.json
│   ├── gui-elements/
│   │   ├── buttons/
│   │   ├── text_fields/
│   │   └── metadata.json
│   └── web-ui/
│       ├── webpage_001.png
│       └── metadata.json
├── annotations/
│   ├── game-ui/
│   │   ├── game-ui_001.json
│   │   └── game-ui_002.json
│   ├── gui-elements/
│   ├── web-ui/
│   ├── class_definitions.json
│   └── split_info.json
└── README.md
```

### B. Common Commands

```bash
# List all datasets
ls -la /home/user/qontinui-finetune/data/datasets/

# Count images in dataset
find /home/user/qontinui-finetune/data/datasets/game-ui -type f -name "*.jpg" | wc -l

# Check disk usage
du -sh /home/user/qontinui-finetune/data/

# Validate dataset
python /home/user/qontinui-finetune/scripts/validate_dataset.py \
  --data-dir /home/user/qontinui-finetune/data/

# View class definitions
cat /home/user/qontinui-finetune/data/annotations/class_definitions.json
```

### C. Data Format Specifications

[TO BE FILLED: Detailed format specifications if needed]

---

## Questions and Support

For questions about data organization or to report issues:

- **Contact:** [TO BE FILLED]
- **Issue tracker:** [TO BE FILLED]
- **Documentation:** See related docs listed above
