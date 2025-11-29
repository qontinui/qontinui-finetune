# Dataset Creation and Preparation Guide

## Overview

This document outlines the strategy, best practices, and step-by-step procedures for creating, organizing, and preparing datasets for fine-tuning vision models on UI detection and segmentation tasks.

---

## 1. Dataset Planning and Strategy

### 1.1 Define Objectives

**Purpose:** [TO BE FILLED]
- Primary use case: [e.g., Desktop UI detection, Mobile UI segmentation]
- Target applications: [List target domains]
- Performance requirements: [Accuracy targets, latency requirements]

### 1.2 Scope Definition

**UI Categories to Cover:**

| Category | Examples | Priority | Status |
|----------|----------|----------|--------|
| Buttons | [TO BE FILLED] | High/Medium/Low | [Pending/In Progress/Complete] |
| Text Fields | [TO BE FILLED] | High/Medium/Low | |
| Menus | [TO BE FILLED] | High/Medium/Low | |
| Dialogs | [TO BE FILLED] | High/Medium/Low | |
| Forms | [TO BE FILLED] | High/Medium/Low | |
| [Other] | [TO BE FILLED] | High/Medium/Low | |

### 1.3 Data Distribution

**Target dataset composition:**
- Total images required: [TO BE FILLED]
- Training set size: [TO BE FILLED] ([Percentage]%)
- Validation set size: [TO BE FILLED] ([Percentage]%)
- Test set size: [TO BE FILLED] ([Percentage]%)

**Class distribution:**
```
[Insert distribution chart or table]
```

---

## 2. Dataset Organization

### 2.1 Directory Structure

Current data directory structure:
```
/home/user/qontinui-finetune/data/
├── datasets/
│   ├── game-ui/              # [TO BE FILLED: Purpose/description]
│   ├── gui-elements/         # [TO BE FILLED: Purpose/description]
│   └── web-ui/               # [TO BE FILLED: Purpose/description]
├── annotations/              # [Annotation files and metadata]
└── README.md                 # This file
```

### 2.2 Recommended Organization Schema

```
/home/user/qontinui-finetune/data/
├── raw/                      # Original, unprocessed images
│   ├── source-1/
│   ├── source-2/
│   └── ...
├── processed/                # Cleaned, preprocessed images
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata/
    ├── class_list.txt
    ├── splits.json
    └── statistics.json
```

### 2.3 Naming Conventions

**Image files:** `[source]_[id]_[variant].jpg`
- Example: `game-ui_001_original.jpg`

**Annotation files:** `[image_id].json` or `[image_id].xml`
- Example: `game-ui_001.json`

**Metadata files:** `[dataset_name]_metadata.json`

---

## 3. Data Collection

### 3.1 Data Sources

**Primary sources for collection:**

| Source | Type | Volume (est.) | Status |
|--------|------|---------------|--------|
| [Source Name] | [Screenshots/API/Manual] | [# images] | [TO BE FILLED] |
| | | | |
| | | | |

**Total collected:** [TO BE FILLED]

### 3.2 Collection Guidelines

- **Image quality:** Minimum [XXX] x [YYY] resolution
- **Format:** PNG, JPG (specify quality settings)
- **Coverage requirements:** [TO BE FILLED]
- **Diversity requirements:** [Different OS/browsers/themes/etc.]

### 3.3 Tools and Methods

**Recommended collection tools:**
```
[TO BE FILLED]
```

**Collection process:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

---

## 4. Image Preprocessing

### 4.1 Preprocessing Steps

**Standard preprocessing pipeline:**

```
Raw Image
    ↓
Resizing
    ↓
Normalization
    ↓
Augmentation (if applicable)
    ↓
Final Dataset Image
```

**Resizing strategy:**
- Target size: [XXX × YYY]
- Aspect ratio handling: [Preserve/Pad/Crop]
- Interpolation method: [TO BE FILLED]

**Normalization:**
- Channel-wise normalization: [Yes/No]
- Mean values: [TO BE FILLED]
- Std values: [TO BE FILLED]

### 4.2 Data Augmentation

**Applied transformations:**

| Transformation | Applied | Range/Intensity | Probability |
|---|---|---|---|
| Rotation | [Yes/No] | [Degrees] | [%] |
| Brightness | [Yes/No] | [Range] | [%] |
| Contrast | [Yes/No] | [Range] | [%] |
| Flip (horizontal) | [Yes/No] | - | [%] |
| Flip (vertical) | [Yes/No] | - | [%] |
| Crop | [Yes/No] | [Range] | [%] |
| Resize | [Yes/No] | [Range] | [%] |
| Noise | [Yes/No] | [Type/Level] | [%] |

**Augmentation configuration location:** [REFERENCE TO: `/home/user/qontinui-finetune/models/configs/`]

### 4.3 Preprocessing Scripts

**Available scripts:** [REFERENCE TO: `/home/user/qontinui-finetune/scripts/`]

**Usage example:**
```bash
python /home/user/qontinui-finetune/scripts/[preprocessing_script.py] \
  --input-dir /home/user/qontinui-finetune/data/raw/ \
  --output-dir /home/user/qontinui-finetune/data/processed/ \
  --target-size 640 640
```

---

## 5. Annotation and Labeling

### 5.1 Annotation Format

**Primary format:** [YOLO/COCO/Pascal VOC/Custom]

**Example annotation structure:**

```json
{
  "image_id": "game-ui_001",
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
      "area": [area_pixels],
      "segmentation": [optional_polygon_coords],
      "iscrowd": 0
    }
  ]
}
```

### 5.2 Class Definitions

**Defined classes:**

| ID | Name | Description | Examples | Notes |
|----|------|-------------|----------|-------|
| 1 | [Class Name] | [Description] | [Examples] | [TO BE FILLED] |
| 2 | | | | |
| | | | | |

**Class definitions file:** [TO BE FILLED or reference to file]

### 5.3 Labeling Process

**Annotation tools:**
- Tool 1: [Tool name and usage notes]
- Tool 2: [Tool name and usage notes]

**Labeling guidelines:**
1. [TO BE FILLED: Specific instructions for annotators]
2. [Boundary definition rules]
3. [Handling of edge cases]

**Quality assurance:**
- Inter-annotator agreement target: [Percentage]
- Review process: [TO BE FILLED]
- Conflict resolution: [TO BE FILLED]

### 5.4 Annotation Statistics

**Annotation Progress:**

| Dataset | Total Images | Annotated | Reviewed | Complete |
|---------|--------------|-----------|----------|----------|
| game-ui | [TO BE FILLED] | [#/Total] | [#/Total] | [Yes/No] |
| gui-elements | [TO BE FILLED] | [#/Total] | [#/Total] | [Yes/No] |
| web-ui | [TO BE FILLED] | [#/Total] | [#/Total] | [Yes/No] |

---

## 6. Dataset Validation

### 6.1 Validation Checks

**Completeness checks:**
- [ ] All images have corresponding annotations
- [ ] All images present in correct directories
- [ ] File naming conventions followed
- [ ] No corrupted files

**Content checks:**
- [ ] Annotation format is valid
- [ ] All objects have valid class IDs
- [ ] Bounding boxes within image bounds
- [ ] Class distribution balanced [or documented]

**Consistency checks:**
- [ ] Consistent image dimensions (or documented variations)
- [ ] Consistent annotation format across dataset
- [ ] No duplicate images

### 6.2 Statistical Analysis

**Dataset statistics to compute:**

```python
# TO BE FILLED: Calculate and document
- Total images: [#]
- Average objects per image: [#]
- Class distribution:
  - Class A: [#] objects ([%])
  - Class B: [#] objects ([%])

- Image dimensions:
  - Min: [X×Y]
  - Max: [X×Y]
  - Average: [X×Y]

- Annotation statistics:
  - Average bbox area: [pixels²]
  - Bbox size range: [min-max pixels²]
```

### 6.3 Validation Script

**Location:** [REFERENCE TO: `/home/user/qontinui-finetune/scripts/validate_dataset.py`]

**Usage:**
```bash
python /home/user/qontinui-finetune/scripts/validate_dataset.py \
  --dataset-dir /home/user/qontinui-finetune/data/processed/ \
  --annotations-dir /home/user/qontinui-finetune/data/annotations/
```

---

## 7. Train/Validation/Test Split

### 7.1 Split Strategy

**Split method:** [Random/Stratified/Manual]

**Split percentages:**
- Training: [TO BE FILLED]%
- Validation: [TO BE FILLED]%
- Test: [TO BE FILLED]%

**Splitting criteria:**
- Ensure no data leakage
- Balanced class distribution across splits
- Representative of real-world scenarios

### 7.2 Split Implementation

**Split configuration file:** `[TO BE FILLED].json`

```json
{
  "strategy": "[random/stratified]",
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "split_by": "[image_id/source]",
  "random_seed": 42,
  "splits": {
    "train": ["image_id_1", "image_id_2", ...],
    "val": [...],
    "test": [...]
  }
}
```

---

## 8. Dataset Documentation

### 8.1 Dataset Metadata

Create `metadata.json` for each dataset:

```json
{
  "dataset_name": "[Name]",
  "dataset_version": "1.0",
  "creation_date": "YYYY-MM-DD",
  "total_images": [#],
  "total_annotations": [#],
  "image_format": "jpg",
  "annotation_format": "coco",
  "classes": {
    "1": "button",
    "2": "text_field"
  },
  "splits": {
    "train": [#],
    "val": [#],
    "test": [#]
  },
  "class_distribution": {
    "button": [#],
    "text_field": [#]
  },
  "sources": ["source1", "source2"],
  "notes": "[Any relevant notes]"
}
```

### 8.2 Creation Log

| Date | Action | Details | Status |
|------|--------|---------|--------|
| YYYY-MM-DD | Collection started | [Details] | [TO BE FILLED] |
| YYYY-MM-DD | Annotation started | [Details] | [TO BE FILLED] |
| YYYY-MM-DD | Validation completed | [Details] | [TO BE FILLED] |

---

## 9. Best Practices

### 9.1 Data Quality

- [ ] Remove duplicates regularly
- [ ] Document and track data source provenance
- [ ] Version control dataset changes
- [ ] Maintain backup copies
- [ ] Document any corrections or modifications

### 9.2 Annotation Quality

- [ ] Use consistent annotation guidelines
- [ ] Perform regular quality reviews
- [ ] Track annotator performance metrics
- [ ] Maintain annotation history

### 9.3 Privacy and Ethics

- [ ] Document data usage rights
- [ ] Ensure compliance with data protection regulations
- [ ] Consider privacy implications of UI screenshots
- [ ] Document any ethical considerations

---

## 10. Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Class imbalance | [Symptom description] | [TO BE FILLED] |
| Poor annotation quality | [Symptom description] | [TO BE FILLED] |
| Data leakage | [Symptom description] | [TO BE FILLED] |
| Corrupted files | [Symptom description] | [TO BE FILLED] |

---

## Related Documentation

- `/home/user/qontinui-finetune/docs/model-selection.md` - Model selection guide
- `/home/user/qontinui-finetune/docs/training-guide.md` - Training workflow
- `/home/user/qontinui-finetune/docs/deployment.md` - Deployment guide
- `/home/user/qontinui-finetune/data/README.md` - Data organization reference

---

## Appendix

### A. Annotation Format Specifications

[TO BE FILLED: Detailed format specifications]

### B. Preprocessing Code Examples

[TO BE FILLED: Code snippets for common preprocessing tasks]

### C. Data Collection Checklist

- [ ] Identify data sources
- [ ] Set up collection infrastructure
- [ ] Collect images
- [ ] Check for quality issues
- [ ] Prepare for annotation
- [ ] Complete annotations
- [ ] Validate dataset
- [ ] Split data
- [ ] Document metadata
