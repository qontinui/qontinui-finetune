# Training Scripts for qontinui-finetune

This directory contains production-ready training scripts for fine-tuning object detection models for GUI element detection.

## Scripts Overview

### 1. prepare_dataset.py
**Purpose**: Dataset preparation and validation

**Features**:
- Format conversion (COCO ↔ YOLO ↔ Pascal VOC)
- Dataset validation and integrity checks
- Train/val/test splitting with stratification
- Comprehensive dataset statistics

**Usage**:
```bash
# Convert COCO to YOLO
python prepare_dataset.py convert \
  --input data/raw \
  --output data/yolo \
  --from-format coco \
  --to-format yolo \
  --coco-json data/raw/annotations.json

# Validate dataset
python prepare_dataset.py validate \
  --dataset data/yolo \
  --format yolo

# Split dataset (70/20/10)
python prepare_dataset.py split \
  --dataset data/yolo \
  --output data/split \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1

# Generate statistics
python prepare_dataset.py stats \
  --dataset data/yolo
```

### 2. train.py
**Purpose**: Model training and fine-tuning

**Features**:
- YOLOv8 support (extensible to Detectron2)
- Comprehensive hyperparameter configuration
- Training monitoring and logging
- Automatic checkpointing
- Resume training support
- Config file support (YAML)

**Usage**:
```bash
# Basic training
python train.py \
  --model yolov8 \
  --model-size n \
  --data data/split/data.yaml \
  --epochs 100 \
  --batch-size 16

# Advanced training with custom hyperparameters
python train.py \
  --model yolov8 \
  --model-size s \
  --data data/split/data.yaml \
  --epochs 200 \
  --batch-size 32 \
  --lr0 0.01 \
  --mosaic 1.0 \
  --mixup 0.15 \
  --device 0 \
  --name gui_detector_v1

# Resume training
python train.py \
  --model yolov8 \
  --data data/split/data.yaml \
  --resume

# Use config file
python train.py --config configs/train_config.yaml
```

### 3. evaluate.py
**Purpose**: Model evaluation and analysis

**Features**:
- mAP@0.5 and mAP@0.5:0.95 calculation
- Precision, Recall, F1 scores
- Per-class performance metrics
- Confusion matrix generation
- Prediction visualizations
- JSON export of results

**Usage**:
```bash
# Basic evaluation
python evaluate.py \
  --model models/yolov8/best.pt \
  --data data/split/data.yaml \
  --output results/eval

# With custom thresholds
python evaluate.py \
  --model models/yolov8/best.pt \
  --data data/split/data.yaml \
  --conf-threshold 0.3 \
  --iou-threshold 0.5 \
  --max-visualizations 50

# Without visualizations (faster)
python evaluate.py \
  --model models/yolov8/best.pt \
  --data data/split/data.yaml \
  --no-visualizations
```

### 4. export.py
**Purpose**: Model export for deployment

**Features**:
- Multiple format support (ONNX, TensorRT, CoreML, TorchScript, OpenVINO)
- Model optimization (FP16, INT8)
- Dynamic shape support
- Post-export validation
- Performance benchmarking

**Usage**:
```bash
# Export to ONNX
python export.py \
  --model models/yolov8/best.pt \
  --format onnx \
  --output exports \
  --imgsz 640 \
  --simplify

# Export to multiple formats
python export.py \
  --model models/yolov8/best.pt \
  --format onnx tensorrt coreml \
  --output exports \
  --validate \
  --benchmark

# Export with optimization
python export.py \
  --model models/yolov8/best.pt \
  --format onnx \
  --dynamic \
  --half \
  --validate
```

### 5. inference.py
**Purpose**: Inference testing and benchmarking

**Features**:
- Multiple model format support (ONNX, PyTorch)
- Single image and batch processing
- Inference speed benchmarking
- Detection visualization
- JSON export of detections
- Real-time display option

**Usage**:
```bash
# Run inference on single image
python inference.py \
  --model exports/best.onnx \
  --input test_images/screenshot.png \
  --output results/inference

# Batch processing
python inference.py \
  --model exports/best.onnx \
  --input test_images/ \
  --output results/inference \
  --save-json

# Benchmark inference speed
python inference.py \
  --model exports/best.onnx \
  --input test_images/screenshot.png \
  --benchmark \
  --benchmark-iterations 100

# With custom thresholds
python inference.py \
  --model exports/best.onnx \
  --input test_images/ \
  --conf-threshold 0.5 \
  --iou-threshold 0.4 \
  --show
```

## Typical Workflow

### 1. Prepare Dataset
```bash
# Validate your dataset
python prepare_dataset.py validate --dataset data/raw

# Split into train/val/test
python prepare_dataset.py split \
  --dataset data/raw \
  --output data/split

# Generate statistics
python prepare_dataset.py stats --dataset data/split/train
```

### 2. Train Model
```bash
# Train YOLOv8 nano model
python train.py \
  --model yolov8 \
  --model-size n \
  --data data/split/data.yaml \
  --epochs 100 \
  --batch-size 16 \
  --name gui_detector_poc
```

### 3. Evaluate Model
```bash
# Evaluate on test set
python evaluate.py \
  --model runs/train/gui_detector_poc/weights/best.pt \
  --data data/split/data.yaml \
  --output results/evaluation
```

### 4. Export Model
```bash
# Export to ONNX for deployment
python export.py \
  --model runs/train/gui_detector_poc/weights/best.pt \
  --format onnx \
  --output exports \
  --validate \
  --benchmark
```

### 5. Test Inference
```bash
# Test exported model
python inference.py \
  --model exports/best.onnx \
  --input test_images/ \
  --output results/inference \
  --benchmark
```

## Script Architecture

All scripts follow consistent patterns:

### Command-Line Interface
- `argparse` for argument parsing
- Comprehensive help messages
- Sensible defaults
- Config file support (where applicable)

### Error Handling
- Input validation
- Graceful error messages
- Exception handling
- File existence checks

### Logging
- Structured logging with timestamps
- Progress bars (tqdm) for long operations
- Summary statistics
- Result visualization

### Type Hints
- Python 3.9+ type hints throughout
- Clear function signatures
- Documented return types

### Documentation
- Module-level docstrings
- Class docstrings
- Function docstrings with Args/Returns
- Inline comments for complex logic

## Configuration

### YOLOv8 Model Sizes

| Size | Params | mAP | Speed (ms) | Recommended Use |
|------|--------|-----|-----------|-----------------|
| n    | 3.2M   | ~   | 1-2       | Development, testing |
| s    | 11.2M  | ~   | 2-3       | Balanced |
| m    | 25.9M  | ~   | 4-5       | Higher accuracy |
| l    | 43.7M  | ~   | 6-8       | Best accuracy |
| x    | 68.2M  | ~   | 10-12     | Maximum accuracy |

### Recommended Settings for GUI Detection

**Small dataset (500-1000 images)**:
```bash
python train.py \
  --model-size n \
  --epochs 100 \
  --batch-size 16 \
  --lr0 0.01 \
  --patience 20
```

**Medium dataset (1000-5000 images)**:
```bash
python train.py \
  --model-size s \
  --epochs 200 \
  --batch-size 32 \
  --lr0 0.01 \
  --mosaic 1.0 \
  --patience 50
```

**Large dataset (5000+ images)**:
```bash
python train.py \
  --model-size m \
  --epochs 300 \
  --batch-size 64 \
  --lr0 0.01 \
  --mosaic 1.0 \
  --mixup 0.15 \
  --patience 100
```

## Performance Targets

Based on AGENT_HANDOFF.md requirements:

- **Accuracy**: mAP@0.5 > 0.90 for common elements
- **Speed**: < 50ms inference time (1080Ti or better)
- **Model size**: < 100MB for deployment
- **Confidence**: > 0.85 for production use

## Dependencies

All scripts require packages from `../requirements.txt`:

```bash
pip install -r ../requirements.txt
```

Key dependencies:
- `ultralytics` - YOLOv8
- `torch` - PyTorch
- `opencv-python` - Image processing
- `onnxruntime` - ONNX inference
- `numpy`, `matplotlib`, `seaborn` - Data processing and visualization

## Extending the Scripts

### Adding New Model Types

1. Create new trainer/evaluator/exporter class
2. Inherit from base class
3. Implement required methods
4. Add to argument parser choices
5. Update documentation

### Adding New Export Formats

1. Add format to `export.py` choices
2. Implement export method in exporter class
3. Add validation method (optional)
4. Update inference.py to support format

### Adding New Metrics

1. Add calculation method to evaluator
2. Update results dictionary
3. Add to summary output
4. Update JSON export

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Use smaller model size (n instead of s)
- Reduce `--imgsz`
- Enable gradient checkpointing

### Slow Training
- Increase `--workers`
- Use GPU (`--device 0`)
- Enable AMP (enabled by default)
- Reduce augmentation complexity

### Poor Accuracy
- Increase dataset size
- Train longer (`--epochs`)
- Use larger model
- Adjust learning rate
- Check dataset quality

### Export Failures
- Update ONNX: `pip install -U onnx onnxruntime`
- For TensorRT: Requires CUDA + TensorRT installed
- For CoreML: Requires macOS or coremltools

## Support

For issues or questions:
1. Check AGENT_HANDOFF.md for project context
2. Review README.md for research prompts
3. Consult ultralytics documentation: https://docs.ultralytics.com/
4. Check GitHub issues in reference repositories

## License

Part of the qontinui project. See main repository for license information.
