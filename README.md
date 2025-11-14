# qontinui-finetune

Fine-tuning existing AI models for GUI element recognition and classification.

## Overview

This repository focuses on adapting pre-trained computer vision models for GUI element detection and classification. The approach leverages transfer learning to build upon existing models rather than training from scratch.

## Goals

1. **Fine-tune existing models** (YOLO, Detectron2, SAM, etc.) for GUI element detection
2. **Support multiple element types**: buttons, text fields, icons, checkboxes, dropdowns, etc.
3. **Handle various GUI frameworks**: desktop (Qt, WPF, Swing), web, mobile
4. **Achieve high accuracy** with limited training data (hundreds to thousands of examples)
5. **Fast inference** for real-time automation (< 100ms per frame)
6. **Export models** in formats compatible with qontinui-runner

## Target Use Cases

- Detecting standard GUI elements (buttons, text fields, icons)
- Recognizing specific patterns (health bars, mini-maps, status indicators)
- Adapting to new applications with minimal retraining
- Supporting qontinui's existing detection strategies

## Model Candidates for Fine-tuning

### Object Detection Models
- **YOLOv8/v9/v10**: Fast, accurate, well-supported for custom training
- **Faster R-CNN**: Higher accuracy, slower inference
- **EfficientDet**: Good balance of speed and accuracy
- **Detectron2**: Facebook's detection platform with many pre-trained models

### Segmentation Models
- **SAM (Segment Anything)**: Meta's powerful segmentation model
- **Mask R-CNN**: Instance segmentation
- **DeepLabv3+**: Semantic segmentation

### Specialized Models
- **UI-BERT**: Pre-trained on UI screenshots
- **Screen2Vec**: Embeddings for UI elements
- **Vision Transformers (ViT)**: Can be adapted for detection

## Research Prompts

### 1. Model Selection Research

**Prompt for initial research:**
```
Research state-of-the-art object detection models suitable for GUI element detection.
Focus on:

1. Models that can be fine-tuned with limited data (transfer learning)
2. Real-time inference performance (< 100ms per frame)
3. Support for bounding box + classification tasks
4. Available pre-trained weights on similar domains
5. Active maintenance and community support
6. Export formats (ONNX, TensorRT, CoreML, etc.)

Compare the following models:
- YOLOv8/v9/v10 (Ultralytics)
- Detectron2 (Facebook)
- EfficientDet
- Faster R-CNN
- SAM (Segment Anything Model)

For each model, provide:
- Architecture overview
- Pre-trained weights available
- Fine-tuning requirements (data, compute)
- Inference speed benchmarks
- Export/deployment options
- Integration complexity
- License considerations

Prioritize models that work well with:
- Small objects (icons, checkboxes)
- Cluttered scenes (complex UIs)
- Variable resolutions
- Limited training data
```

### 2. Dataset Creation Strategy

**Prompt:**
```
Design a dataset creation strategy for GUI element detection training. Consider:

1. Data sources:
   - Synthetic UI generation
   - Screenshot scraping with auto-labeling
   - Existing datasets (Rico, CLAY, UIBert)
   - Game-specific screenshots (for gaming automation)
   - Manual annotation tools

2. Annotation format:
   - Bounding boxes with element type labels
   - Hierarchical classifications (element type, subtype, state)
   - Additional attributes (text content, icons, states)
   - Compatibility with COCO, YOLO, Pascal VOC formats

3. Data augmentation:
   - Resolution changes
   - Color/brightness variations
   - Rotation and scaling
   - Occlusion simulation
   - Background variations

4. Quality assurance:
   - Annotation validation
   - Inter-annotator agreement
   - Edge case coverage
   - Balance across element types

Provide specific recommendations for:
- Minimum dataset size per element type
- Train/val/test splits
- Handling class imbalance
- Incremental dataset growth
```

### 3. Fine-tuning Pipeline Design

**Prompt:**
```
Design a complete fine-tuning pipeline for GUI element detection:

1. Data preprocessing:
   - Image normalization
   - Resizing strategies
   - Annotation format conversion
   - Data validation

2. Training configuration:
   - Learning rate schedules
   - Batch sizes for different GPUs
   - Augmentation strategies
   - Loss functions
   - Regularization techniques

3. Monitoring and evaluation:
   - Metrics (mAP, precision, recall, F1)
   - Validation strategies
   - Early stopping criteria
   - Model checkpointing

4. Hyperparameter optimization:
   - Grid search vs random search
   - Learning rate tuning
   - Architecture variations
   - Data augmentation parameters

5. Model export and optimization:
   - ONNX conversion
   - TensorRT optimization
   - Quantization (INT8, FP16)
   - Model pruning

Provide code structure and best practices.
```

### 4. Open Source Repository Research

**Prompt:**
```
Find and analyze open source repositories for GUI/UI element detection:

Search for repositories that:
1. Fine-tune object detection models on UI/GUI datasets
2. Provide annotation tools for GUI elements
3. Implement UI understanding models
4. Include pre-trained models for screen understanding
5. Offer deployment pipelines for CV models

Key repositories to research:
- Ultralytics YOLOv8 (object detection)
- Detectron2 (Facebook's detection platform)
- MMDetection (OpenMMLab toolkit)
- Screen Recognition (UI automation projects)
- Rico/CLAY dataset implementations
- UI-BERT and similar UI understanding models

For each repository:
- URL and description
- Relevance to GUI detection
- Code quality and documentation
- Reusable components
- License
- Active maintenance status
- Integration potential with qontinui

Prioritize repositories with:
- Clean, modular code
- Good documentation
- Training scripts
- Inference examples
- Model zoo with pre-trained weights
```

### 5. Integration with Qontinui

**Prompt:**
```
Design the integration between fine-tuned models and qontinui-runner:

1. Model format and export:
   - ONNX for cross-platform compatibility
   - TensorRT for GPU optimization
   - CoreML for macOS
   - OpenVINO for Intel CPUs

2. Inference API:
   - Input: screenshot (numpy array or PIL Image)
   - Output: List of detections (bbox, class, confidence)
   - Batch processing support
   - GPU/CPU fallback

3. Configuration:
   - Model selection (multiple models for different tasks)
   - Confidence thresholds
   - NMS parameters
   - Input preprocessing

4. Performance optimization:
   - Model caching
   - Batch inference
   - Async processing
   - Memory management

5. Fallback strategies:
   - Combine ML detection with traditional CV
   - Multi-model ensembles
   - Confidence-based switching

Provide Python interface design and usage examples.
```

## Repository Structure

```
qontinui-finetune/
├── README.md                          # This file
├── RESEARCH.md                        # Research findings and decisions
├── docs/
│   ├── model-selection.md            # Model comparison and selection
│   ├── dataset-creation.md           # Dataset strategy
│   ├── training-guide.md             # Training workflow
│   └── deployment.md                 # Export and deployment
├── data/
│   ├── datasets/                     # Training datasets
│   │   ├── gui-elements/            # General GUI elements
│   │   ├── game-ui/                 # Game-specific UI
│   │   └── web-ui/                  # Web application UI
│   ├── annotations/                  # Annotation files
│   └── README.md                     # Dataset documentation
├── models/
│   ├── yolov8/                      # YOLO fine-tuning
│   ├── detectron2/                  # Detectron2 fine-tuning
│   ├── sam/                         # SAM fine-tuning
│   └── configs/                     # Model configurations
├── scripts/
│   ├── prepare_dataset.py           # Dataset preparation
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Evaluation script
│   ├── export.py                    # Model export (ONNX, etc.)
│   └── inference.py                 # Inference testing
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Dataset analysis
│   ├── 02_training.ipynb            # Training experiments
│   └── 03_evaluation.ipynb          # Results analysis
├── reference/                        # Cloned reference repos
│   └── README.md                    # Reference repo list
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- 8GB+ VRAM (for training)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e ".[dev]"
```

### Quick Start

1. **Prepare dataset**:
```bash
python scripts/prepare_dataset.py --source data/raw --output data/processed
```

2. **Train model**:
```bash
python scripts/train.py --model yolov8 --data data/processed --epochs 100
```

3. **Evaluate**:
```bash
python scripts/evaluate.py --model models/best.pt --data data/processed/val
```

4. **Export for deployment**:
```bash
python scripts/export.py --model models/best.pt --format onnx
```

## Dataset Requirements

### Element Types to Detect

Based on qontinui's existing strategies:

1. **Interactive Elements**:
   - Buttons (standard, icon, toggle)
   - Text fields (input, search, password)
   - Checkboxes
   - Radio buttons
   - Dropdowns/Select menus
   - Sliders
   - Switches/Toggles

2. **Display Elements**:
   - Labels/Text
   - Images
   - Icons
   - Progress bars
   - Status indicators

3. **Container Elements**:
   - Panels/Cards
   - Tabs
   - Menus
   - Dialogs/Modals
   - Tooltips

4. **Game-Specific**:
   - Health/Mana bars
   - Mini-maps
   - Inventory slots
   - Action bars
   - Chat windows

### Minimum Dataset Sizes

- **Initial training**: 500-1000 examples per element type
- **Production quality**: 2000-5000 examples per element type
- **Validation set**: 20% of training data
- **Test set**: 10% of total data

## Performance Targets

- **Accuracy**: mAP@0.5 > 0.90 for common elements
- **Speed**: < 50ms inference time (1080Ti or better)
- **Model size**: < 100MB for deployment
- **Confidence**: > 0.85 for production use

## Contributing

1. Follow existing code structure
2. Document all experiments in notebooks
3. Update RESEARCH.md with findings
4. Add tests for new functionality
5. Follow Python best practices (PEP 8, type hints)

## License

TBD - Align with qontinui project license

## Related Projects

- [qontinui-runner](../qontinui-runner): The main automation engine
- [qontinui-train](../qontinui-train): Training models from scratch
- [qontinui-web](../qontinui-web): Web interface for management
