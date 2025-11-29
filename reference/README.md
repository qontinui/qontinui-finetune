# Reference Repositories for Fine-tuning

This directory contains cloned reference repositories for learning and adapting code for GUI element detection fine-tuning.

## Recommended Repositories to Clone

### Object Detection Frameworks

1. **Ultralytics YOLOv8**
   - URL: https://github.com/ultralytics/ultralytics
   - Purpose: State-of-the-art YOLO implementation with excellent fine-tuning support
   - Key features: Easy custom dataset training, ONNX export, comprehensive docs
   - Clone command: `git clone https://github.com/ultralytics/ultralytics.git`

2. **Detectron2**
   - URL: https://github.com/facebookresearch/detectron2
   - Purpose: Facebook's detection platform with many pre-trained models
   - Key features: Mask R-CNN, Faster R-CNN, extensive model zoo
   - Clone command: `git clone https://github.com/facebookresearch/detectron2.git`

3. **MMDetection**
   - URL: https://github.com/open-mmlab/mmdetection
   - Purpose: OpenMMLab's comprehensive detection toolbox
   - Key features: 50+ detection models, modular design, good documentation
   - Clone command: `git clone https://github.com/open-mmlab/mmdetection.git`

### Segmentation Models

4. **Segment Anything (SAM)**
   - URL: https://github.com/facebookresearch/segment-anything
   - Purpose: Meta's universal segmentation model
   - Key features: Prompt-based segmentation, fine-tuning examples
   - Clone command: `git clone https://github.com/facebookresearch/segment-anything.git`

5. **Grounded-SAM**
   - URL: https://github.com/IDEA-Research/Grounded-Segment-Anything
   - Purpose: SAM + Grounding DINO for text-prompted detection
   - Key features: Open-vocabulary detection, SAM integration
   - Clone command: `git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git`

### UI-Specific Research

6. **Screen Recognition**
   - URL: https://github.com/google-research/google-research/tree/master/screen_recognition
   - Purpose: Google's screen understanding research
   - Key features: Screen element detection, UI understanding
   - Clone command: `git clone https://github.com/google-research/google-research.git` (then navigate to screen_recognition)

7. **UIBert**
   - URL: https://github.com/google-research/google-research/tree/master/uibert
   - Purpose: BERT for UI understanding
   - Key features: Pre-trained on UI data, multi-modal
   - Clone command: Part of google-research repo

8. **Rico Dataset Tools**
   - URL: https://github.com/google-research-datasets/rico
   - Purpose: Tools for working with Rico mobile UI dataset
   - Key features: Dataset loading, visualization
   - Clone command: `git clone https://github.com/google-research-datasets/rico.git`

### Training Utilities

9. **PyTorch Lightning**
   - URL: https://github.com/Lightning-AI/pytorch-lightning
   - Purpose: High-level PyTorch training framework
   - Key features: Simplified training loops, multi-GPU support
   - Clone command: `git clone https://github.com/Lightning-AI/pytorch-lightning.git`

10. **timm (PyTorch Image Models)**
    - URL: https://github.com/huggingface/pytorch-image-models
    - Purpose: Pre-trained vision models library
    - Key features: 1000+ models, transfer learning utilities
    - Clone command: `git clone https://github.com/huggingface/pytorch-image-models.git`

### Data Augmentation

11. **Albumentations**
    - URL: https://github.com/albumentations-team/albumentations
    - Purpose: Fast image augmentation library
    - Key features: Detection-aware augmentations, GPU support
    - Clone command: `git clone https://github.com/albumentations-team/albumentations.git`

### Annotation Tools

12. **CVAT**
    - URL: https://github.com/opencv/cvat
    - Purpose: Computer Vision Annotation Tool
    - Key features: Web-based, semi-automatic annotation
    - Clone command: `git clone https://github.com/opencv/cvat.git`

13. **Label Studio**
    - URL: https://github.com/HumanSignal/label-studio
    - Purpose: Multi-type data labeling tool
    - Key features: ML-assisted labeling, flexible configuration
    - Clone command: `git clone https://github.com/HumanSignal/label-studio.git`

## How to Use These References

1. **Clone to this directory**:
   ```bash
   cd reference/
   git clone <repository-url>
   ```

2. **Study the code structure**:
   - Training scripts and pipelines
   - Data loading and preprocessing
   - Model architectures
   - Evaluation metrics
   - Export and deployment

3. **Adapt for GUI detection**:
   - Modify data loaders for GUI datasets
   - Adjust architectures for GUI-specific patterns
   - Customize augmentation strategies
   - Add GUI-specific evaluation metrics

4. **Extract reusable components**:
   - Copy and adapt utilities
   - Reference training recipes
   - Learn best practices
   - Avoid common pitfalls

## Git Ignore

Add to `.gitignore` to avoid committing reference repos:
```
reference/*/
!reference/README.md
```

## License Considerations

- Check each repository's license before using code
- Attribute properly when adapting code
- Ensure compatibility with qontinui's license
- Follow open source best practices

## Maintenance

- Periodically update reference repos: `git pull` in each directory
- Track which commits/versions are being referenced
- Document specific code adaptations in RESEARCH.md
