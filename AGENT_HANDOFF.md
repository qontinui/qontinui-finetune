# Agent Handoff - qontinui-finetune

This document provides context for AI agents continuing work on this repository.

## Project Context

**Repository**: qontinui-finetune
**Purpose**: Fine-tuning existing pre-trained models for GUI element detection
**Owner**: Joshua Spinak
**Organization**: qontinui
**GitHub**: https://github.com/qontinui/qontinui-finetune.git

## What This Repository Does

This repository focuses on **fine-tuning existing computer vision models** (like YOLOv8, Detectron2, SAM) to detect GUI elements in screenshots. The approach uses transfer learning to build upon pre-trained models rather than training from scratch.

### Target Use Cases:
- Detecting standard GUI elements (buttons, text fields, icons, checkboxes, dropdowns)
- Recognizing game-specific UI patterns (health bars, mini-maps, status indicators)
- Adapting to new applications with minimal retraining (hundreds to thousands of examples)
- Supporting qontinui's existing automation engine (qontinui-runner)

### Key Differentiators:
- **Fast training**: Hours to days on single GPU
- **Limited data**: 500-5000 examples per element type
- **Transfer learning**: Leverages pre-trained weights
- **Production-ready**: Export to ONNX, TensorRT for deployment

## Current State

### ✅ Completed:
1. Repository structure created with all directories
2. Comprehensive README with 5 detailed research prompts
3. Reference repository guide (13 repos to clone for learning)
4. requirements.txt with all dependencies
5. .gitignore configured for ML projects
6. Git initialized with `main` branch
7. Remote configured: https://github.com/qontinui/qontinui-finetune.git
8. Pre-commit hook installed to prevent Claude attribution

### 📋 Not Yet Done:
- Clone reference repositories to `reference/` directory
- Execute research prompts to select model architecture
- Create initial dataset (even small proof-of-concept)
- Implement training pipeline
- Create evaluation scripts
- Export and test models

## Important Files

1. **README.md**: Complete project overview with research prompts
2. **reference/README.md**: Guide to 13 reference repositories to clone
3. **requirements.txt**: All Python dependencies
4. **.gitignore**: Configured to exclude models, datasets, logs
5. **.git/hooks/commit-msg**: Pre-commit hook preventing Claude attribution

## Research Prompts Available

The README contains 5 detailed research prompts. **Use these first** before writing code:

1. **Model Selection Research**: Compare YOLOv8, Detectron2, SAM, EfficientDet, Faster R-CNN
2. **Dataset Creation Strategy**: Synthetic generation, annotation tools, data augmentation
3. **Fine-tuning Pipeline Design**: Training config, hyperparameters, monitoring
4. **Open Source Repository Research**: Find and analyze relevant codebases
5. **Integration with Qontinui**: Design inference API for qontinui-runner

## Recommended Next Steps

### Phase 1: Research (Week 1)
1. **Execute Model Selection prompt** to compare architectures
2. **Clone reference repositories**:
   ```bash
   cd reference/
   git clone https://github.com/ultralytics/ultralytics.git
   git clone https://github.com/facebookresearch/detectron2.git
   git clone https://github.com/facebookresearch/segment-anything.git
   ```
3. **Study YOLOv8 first** (easiest to get started, good documentation)
4. **Document findings** in `RESEARCH.md`

### Phase 2: Proof of Concept (Week 2)
1. **Create small test dataset** (100-200 annotated screenshots):
   - Use Label Studio or CVAT for annotation
   - Focus on 3-5 element types (buttons, text fields, icons)
   - Store in `data/datasets/poc/`

2. **Fine-tune YOLOv8**:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')  # Start with nano model
   results = model.train(data='data.yaml', epochs=100)
   ```

3. **Create evaluation script** in `scripts/evaluate.py`

4. **Test export**:
   ```python
   model.export(format='onnx')
   ```

### Phase 3: Scaling (Weeks 3-4)
1. Expand dataset to 500-1000 examples per class
2. Try other architectures (Detectron2, SAM)
3. Optimize hyperparameters
4. Benchmark performance (mAP, speed, size)
5. Create deployment guide

## Integration Points

### With qontinui-runner:
- Export models to ONNX format
- Provide Python inference API
- Document input/output format
- Specify confidence thresholds

### With qontinui-web:
- Dataset management interface
- Training job monitoring
- Model performance metrics
- A/B testing different models

## Technical Constraints

### Performance Targets:
- **Accuracy**: mAP@0.5 > 0.90 for common elements
- **Speed**: < 50ms inference time (1080Ti or better)
- **Model size**: < 100MB for deployment
- **Confidence**: > 0.85 for production use

### Dataset Requirements:
- **Minimum**: 500 examples per element type
- **Recommended**: 2000-5000 examples per element type
- **Formats**: COCO, YOLO, Pascal VOC
- **Splits**: 70% train, 20% val, 10% test

## Code Quality Standards

Follow existing qontinui conventions:
- Python 3.9+ with type hints
- Black formatting
- Comprehensive docstrings
- Unit tests for utilities
- Clear logging
- No backward compatibility needed (active development)

## Commit Guidelines

**CRITICAL**: This repo has a pre-commit hook that will **reject commits** containing:
- "Co-Authored-By: Claude"
- "Generated with [Claude Code]"
- Robot emoji (🤖)

Only Joshua Spinak should be credited as the author.

## Key Decision Points

When working on this repo, you'll need to make these decisions:

1. **Which model architecture?**
   - YOLOv8 (fast, easy, good docs) ← Recommended for starting
   - Detectron2 (more accurate, more complex)
   - SAM (segmentation-first approach)

2. **Dataset strategy?**
   - Manual annotation (Label Studio, CVAT)
   - Synthetic generation (programmatic UI creation)
   - Existing datasets (Rico for mobile, custom for desktop/games)

3. **Training infrastructure?**
   - Local GPU (RTX 3080+)
   - Cloud (AWS/GCP/Azure p3/p4 instances)
   - Colab/Kaggle (free tier for prototyping)

4. **Export format?**
   - ONNX (cross-platform)
   - TensorRT (NVIDIA optimization)
   - CoreML (macOS)
   - Multiple formats

## Related Projects

- **qontinui-runner** (`../qontinui-runner/`): Main automation engine that will use these models
- **qontinui-train** (`../qontinui-train/`): Sister repo for training from scratch (larger scale)
- **qontinui-web** (`../qontinui-web/`): Web interface for managing projects

## Resources

### Documentation:
- YOLOv8: https://docs.ultralytics.com/
- Detectron2: https://detectron2.readthedocs.io/
- SAM: https://github.com/facebookresearch/segment-anything

### Datasets:
- Rico (mobile UI): https://interactionmining.org/rico
- CLAY: https://github.com/google-research-datasets/clay
- UI-BERT datasets: Check google-research repos

### Papers:
- YOLOv8: https://arxiv.org/abs/2305.09972
- SAM: https://arxiv.org/abs/2304.02643
- Screen Recognition: Search for UI understanding papers

## Communication

When updating this repository:
1. Document decisions in `RESEARCH.md`
2. Update README if approach changes
3. Add training recipes to `models/configs/`
4. Create notebooks for experiments
5. Keep detailed logs of training runs

## Questions to Answer

As you work, try to answer these questions:

1. **Model Selection**: Which architecture achieves the best balance of accuracy and speed for GUI detection?
2. **Data Requirements**: How much data is actually needed for production-quality detection?
3. **Augmentation**: What augmentation strategies work best for UI images?
4. **Class Imbalance**: How to handle rare element types?
5. **Cross-platform**: How well do models generalize across different OS/applications?
6. **Deployment**: What's the most efficient export format for qontinui-runner?

## Example Workflow

Here's a complete workflow from start to deployment:

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Clone reference repo
cd reference/
git clone https://github.com/ultralytics/ultralytics.git

# 3. Create dataset
# Use Label Studio to annotate 100 screenshots
# Export in YOLO format to data/datasets/poc/

# 4. Train model
cd ../scripts/
python train.py --model yolov8n --data ../data/datasets/poc/data.yaml --epochs 100

# 5. Evaluate
python evaluate.py --model ../models/yolov8/best.pt --data ../data/datasets/poc/val/

# 6. Export
python export.py --model ../models/yolov8/best.pt --format onnx

# 7. Test inference
python inference.py --model ../models/yolov8/best.onnx --image test.png
```

## Success Criteria

You'll know the project is successful when:
- ✅ Model achieves >90% mAP on test set
- ✅ Inference runs in <50ms per frame
- ✅ Model size is <100MB
- ✅ Exported model works in qontinui-runner
- ✅ Documentation is complete
- ✅ Other developers can reproduce results

## Getting Help

If you need more context:
1. Read the complete README.md (has all research prompts)
2. Check reference/README.md (13 repos to learn from)
3. Look at qontinui-runner to understand integration needs
4. Review qontinui-web to understand data management

## Final Notes

- **Start small**: Don't try to solve everything at once
- **Document everything**: Future you will thank you
- **Test early**: Don't wait until the end to test inference
- **Iterate quickly**: Fast experiments > perfect first try
- **Ask questions**: Better to clarify than assume

Good luck! This is an exciting project with real-world impact on GUI automation.
