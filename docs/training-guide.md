# Training Guide: Step-by-Step Workflow

## Overview

This guide provides comprehensive step-by-step instructions for training vision models on the prepared datasets. It covers model selection, hyperparameter configuration, training execution, and validation.

---

## 1. Pre-Training Checklist

### 1.1 Environment Setup

- [ ] CUDA/GPU drivers installed
  - Required version: [TO BE FILLED]
  - Verify: `nvidia-smi`

- [ ] Python environment created
  ```bash
  python --version  # Should be 3.8+
  ```

- [ ] Dependencies installed
  ```bash
  pip install -r /home/user/qontinui-finetune/requirements.txt
  ```

- [ ] GPU availability confirmed
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

### 1.2 Data Preparation

- [ ] Dataset located: `/home/user/qontinui-finetune/data/`
- [ ] All images present and accessible
- [ ] Annotations validated
  ```bash
  # Validation script location: [TO BE FILLED]
  ```

- [ ] Train/Val/Test splits created
- [ ] Dataset statistics computed

### 1.3 Model Selection

Selected model: [TO BE FILLED]
- Model location: `/home/user/qontinui-finetune/models/[model_name]/`
- Configuration file: [TO BE FILLED]

---

## 2. Configuration Setup

### 2.1 Model Configuration

**Configuration file location:** `/home/user/qontinui-finetune/models/configs/`

**Configuration structure:**

```yaml
# model-config.yaml [TO BE FILLED]
model:
  type: "[yolo/sam/detectron2]"
  variant: "[nano/small/medium/large/xlarge]"
  pretrained: true
  pretrained_weights: "[path/to/weights]"

input:
  size: [640, 640]  # [width, height]
  channels: 3
  normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

data:
  num_classes: [TO BE FILLED]
  classes: [TO BE FILLED]  # ["button", "textfield", ...]
```

### 2.2 Training Configuration

**Training hyperparameters:**

```yaml
training:
  epochs: [TO BE FILLED]
  batch_size: [TO BE FILLED]
  learning_rate: [TO BE FILLED]
  optimizer: [TO BE FILLED]  # adam, sgd, adamw
  loss_function: [TO BE FILLED]

  # Learning rate scheduler
  scheduler:
    type: [TO BE FILLED]  # cosine, linear, step
    warmup_epochs: [TO BE FILLED]

  # Early stopping
  early_stopping:
    enabled: [true/false]
    patience: [TO BE FILLED]
    metric: "val_loss"

  # Mixed precision training
  mixed_precision: [true/false]

  # Gradient accumulation
  gradient_accumulation_steps: 1

validation:
  frequency: 1  # Validate every N epochs
  metrics:
    - mAP
    - mAP50
    - mAP75
    - F1

augmentation:
  enabled: true
  transformations:
    - flip_horizontal
    - flip_vertical
    - rotation: [0, 15]
    - brightness: [0.8, 1.2]
    - contrast: [0.8, 1.2]
```

**Save configuration to:** `/home/user/qontinui-finetune/models/configs/training_config.yaml`

### 2.3 Environment Variables

```bash
# Set project paths
export QONTINUI_ROOT=/home/user/qontinui-finetune
export DATA_DIR=$QONTINUI_ROOT/data
export MODEL_DIR=$QONTINUI_ROOT/models
export LOGS_DIR=$QONTINUI_ROOT/logs

# GPU settings
export CUDA_VISIBLE_DEVICES=0  # Or 0,1,2,... for multiple GPUs
export CUDA_LAUNCH_BLOCKING=1  # For debugging

# Logging
export LOG_LEVEL=INFO
```

---

## 3. Training Execution

### 3.1 Model-Specific Training

#### Option A: YOLO v8

**Training script location:** `/home/user/qontinui-finetune/scripts/train_yolo.py` [TO BE FILLED]

**Basic training command:**
```bash
python /home/user/qontinui-finetune/scripts/train_yolo.py \
  --config /home/user/qontinui-finetune/models/configs/yolo_config.yaml \
  --data-dir /home/user/qontinui-finetune/data/processed/ \
  --output-dir /home/user/qontinui-finetune/outputs/yolo_run_1 \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.001
```

**Advanced options:**
```bash
python /home/user/qontinui-finetune/scripts/train_yolo.py \
  --config /home/user/qontinui-finetune/models/configs/yolo_config.yaml \
  --data-dir /home/user/qontinui-finetune/data/processed/ \
  --output-dir /home/user/qontinui-finetune/outputs/yolo_run_1 \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --resume-from [checkpoint_path] \
  --mixed-precision \
  --num-workers 4 \
  --seed 42 \
  --verbose
```

#### Option B: SAM (Segment Anything Model)

**Training script location:** `/home/user/qontinui-finetune/scripts/train_sam.py` [TO BE FILLED]

**Basic training command:**
```bash
python /home/user/qontinui-finetune/scripts/train_sam.py \
  --config /home/user/qontinui-finetune/models/configs/sam_config.yaml \
  --data-dir /home/user/qontinui-finetune/data/processed/ \
  --output-dir /home/user/qontinui-finetune/outputs/sam_run_1 \
  --epochs 50 \
  --batch-size 8
```

**Key considerations for SAM:**
- [TO BE FILLED: SAM-specific training notes]

#### Option C: Detectron2

**Training script location:** `/home/user/qontinui-finetune/scripts/train_detectron2.py` [TO BE FILLED]

**Basic training command:**
```bash
python /home/user/qontinui-finetune/scripts/train_detectron2.py \
  --config /home/user/qontinui-finetune/models/configs/detectron2_config.yaml \
  --data-dir /home/user/qontinui-finetune/data/processed/ \
  --output-dir /home/user/qontinui-finetune/outputs/detectron2_run_1 \
  --epochs 100 \
  --batch-size 16
```

### 3.2 Distributed Training

**For multi-GPU training:**

```bash
# Using DataParallel
python /home/user/qontinui-finetune/scripts/train_[model].py \
  --multi-gpu \
  --gpu-ids 0,1,2,3 \
  [other options]

# OR using DistributedDataParallel
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  /home/user/qontinui-finetune/scripts/train_[model].py \
  [options]
```

### 3.3 Monitoring Training

**TensorBoard:**
```bash
tensorboard --logdir=/home/user/qontinui-finetune/outputs/[run_name]/logs
```

**Weights & Biases (if configured):**
```bash
# Logs will appear at: https://wandb.ai/[username]/qontinui-finetune
```

**Key metrics to monitor:**
- Training loss
- Validation loss
- [Metric 1]: [Target value]
- [Metric 2]: [Target value]
- Learning rate schedule
- GPU memory usage

---

## 4. Training Iterations and Debugging

### 4.1 Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Out of memory (OOM) | CUDA out of memory error | Reduce batch size, use gradient accumulation |
| Poor loss decrease | Loss plateaus or increases | Adjust learning rate, check data quality |
| Overfitting | High train accuracy, low val accuracy | Add regularization, more augmentation, more data |
| Underfitting | Both train and val loss high | Increase model capacity, more training steps |
| NaN loss | Loss becomes NaN during training | Check data normalization, reduce learning rate |

### 4.2 Hyperparameter Tuning

**Systematic tuning approach:**

1. **Learning rate search** (1-2 epochs)
   - Start with LR finder: `learning_rate ∈ [1e-5, 1e-1]`
   - Select LR where loss decreases fastest

2. **Batch size optimization**
   - Test values: 8, 16, 32, 64
   - Larger batches often improve generalization

3. **Warmup and scheduler tuning**
   - Warmup epochs: [TO BE FILLED]
   - Decay strategy: [TO BE FILLED]

4. **Model capacity (architecture)**
   - Start with [TO BE FILLED] size model
   - Scale up if underfitting, down if overfitting

**Tuning log template:**

| Run | Epoch | LR | Batch | Val mAP | Notes | Status |
|-----|-------|----|----|---------|-------|--------|
| 1 | 50 | 0.001 | 16 | [Value] | Initial baseline | Complete |
| 2 | 50 | 0.0005 | 16 | [Value] | [Notes] | Complete |
| 3 | 100 | 0.001 | 32 | [Value] | [Notes] | Pending |

---

## 5. Checkpoint Management

### 5.1 Saving Checkpoints

**Checkpoint structure:**

```
outputs/[run_name]/
├── checkpoints/
│   ├── epoch_001.pt
│   ├── epoch_010.pt
│   ├── epoch_020.pt
│   ├── best_model.pt          # Best validation performance
│   └── latest_model.pt         # Latest epoch
├── logs/
│   ├── training_log.csv
│   └── tensorboard_events/
├── config.yaml                 # Training configuration
└── metrics.json                # Final metrics
```

### 5.2 Resuming Training

**Resume from specific checkpoint:**
```bash
python /home/user/qontinui-finetune/scripts/train_[model].py \
  --config /home/user/qontinui-finetune/models/configs/[model]_config.yaml \
  --resume-from /home/user/qontinui-finetune/outputs/[run_name]/checkpoints/epoch_050.pt \
  --epochs 150 \
  [other options]
```

### 5.3 Best Model Selection

**Criteria for best model:**
1. Highest validation mAP on [TO BE FILLED] dataset
2. Reasonable inference speed ([TO BE FILLED] ms)
3. No signs of overfitting

**Best model location:** `/home/user/qontinui-finetune/outputs/[run_name]/checkpoints/best_model.pt`

---

## 6. Evaluation and Validation

### 6.1 Validation Metrics

**Primary metrics:**

| Metric | Formula | Target | Notes |
|--------|---------|--------|-------|
| mAP (mean Average Precision) | Average of AP@IoU across classes | [TO BE FILLED] | Standard evaluation metric |
| mAP50 | AP@IoU=0.5 | [TO BE FILLED] | Loose bounding box criterion |
| mAP75 | AP@IoU=0.75 | [TO BE FILLED] | Strict bounding box criterion |
| Precision | TP / (TP + FP) | [TO BE FILLED] | False positive rate |
| Recall | TP / (TP + FN) | [TO BE FILLED] | False negative rate |
| F1 Score | 2 × (Precision × Recall) / (Precision + Recall) | [TO BE FILLED] | Harmonic mean |

**Per-class metrics:**
- [Class 1]: mAP = [Target]
- [Class 2]: mAP = [Target]

### 6.2 Validation Script

**Location:** `/home/user/qontinui-finetune/scripts/evaluate.py` [TO BE FILLED]

**Usage:**
```bash
python /home/user/qontinui-finetune/scripts/evaluate.py \
  --model-path /home/user/qontinui-finetune/outputs/[run_name]/checkpoints/best_model.pt \
  --config /home/user/qontinui-finetune/models/configs/[model]_config.yaml \
  --data-dir /home/user/qontinui-finetune/data/processed/ \
  --split test \
  --output-dir /home/user/qontinui-finetune/outputs/[run_name]/evaluation
```

### 6.3 Visualization

**Generate evaluation visualizations:**

```bash
python /home/user/qontinui-finetune/scripts/visualize_results.py \
  --predictions /home/user/qontinui-finetune/outputs/[run_name]/evaluation/predictions.json \
  --images-dir /home/user/qontinui-finetune/data/processed/test \
  --output-dir /home/user/qontinui-finetune/outputs/[run_name]/visualizations \
  --num-samples 20
```

---

## 7. Final Training Report

### 7.1 Training Summary

**Template to fill in:**

```markdown
## Training Summary

- **Model:** [Model name and variant]
- **Dataset:** [Dataset names and sizes]
- **Training Date:** YYYY-MM-DD to YYYY-MM-DD
- **Total Epochs:** [#]
- **Best Epoch:** [#]
- **Best Checkpoint:** [filename]

### Final Metrics

- **Validation mAP:** [Value]
- **Validation mAP50:** [Value]
- **Validation mAP75:** [Value]
- **Test mAP:** [Value] (if available)

### Hardware Used

- **GPU(s):** [Model and count]
- **Total Training Time:** [Hours/Days]
- **Peak GPU Memory:** [GB]

### Key Findings

1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### Recommendations for Next Steps

1. [Recommendation 1]
2. [Recommendation 2]
```

### 7.2 Logging Training Results

**Save to:** `/home/user/qontinui-finetune/logs/training_results.json`

```json
{
  "experiment_name": "[name]",
  "timestamp": "2024-MM-DDTHH:MM:SS",
  "model": {
    "type": "[model_type]",
    "variant": "[variant]"
  },
  "dataset": {
    "names": ["[dataset1]", "[dataset2]"],
    "train_size": [#],
    "val_size": [#],
    "test_size": [#]
  },
  "training": {
    "epochs": [#],
    "batch_size": [#],
    "learning_rate": [value],
    "total_time_hours": [value]
  },
  "results": {
    "best_val_mAP": [value],
    "best_val_mAP50": [value],
    "test_mAP": [value]
  }
}
```

---

## 8. Troubleshooting Reference

### 8.1 Debugging Mode

**Run training with verbose output:**
```bash
python /home/user/qontinui-finetune/scripts/train_[model].py \
  --verbose \
  --log-level DEBUG \
  [other options]
```

### 8.2 Data Issues

```bash
# Verify dataset structure
python /home/user/qontinui-finetune/scripts/check_dataset.py \
  --data-dir /home/user/qontinui-finetune/data/processed/

# Visualize batch
python /home/user/qontinui-finetune/scripts/visualize_batch.py \
  --data-dir /home/user/qontinui-finetune/data/processed/ \
  --batch-size 4
```

### 8.3 Getting Help

- **Common issues:** [TO BE FILLED: Link to FAQ or issues document]
- **Log file location:** `/home/user/qontinui-finetune/logs/training_[timestamp].log`
- **Issue tracking:** [TO BE FILLED: Github issues or internal tracking]

---

## Related Documentation

- `/home/user/qontinui-finetune/docs/model-selection.md` - Model selection guide
- `/home/user/qontinui-finetune/docs/dataset-creation.md` - Dataset preparation
- `/home/user/qontinui-finetune/docs/deployment.md` - Deployment guide
- `/home/user/qontinui-finetune/README.md` - Project overview
- `/home/user/qontinui-finetune/requirements.txt` - Dependencies

---

## Appendix: Quick Reference

### Command Cheat Sheet

```bash
# Environment setup
export CUDA_VISIBLE_DEVICES=0
pip install -r requirements.txt

# Start training
python scripts/train_[model].py \
  --config models/configs/[model]_config.yaml \
  --data-dir data/processed/ \
  --output-dir outputs/run_1

# Monitor with TensorBoard
tensorboard --logdir=outputs/run_1/logs

# Evaluate
python scripts/evaluate.py \
  --model-path outputs/run_1/checkpoints/best_model.pt \
  --data-dir data/processed/

# Resume training
python scripts/train_[model].py \
  --resume-from outputs/run_1/checkpoints/epoch_050.pt
```

### Expected Training Times

| Model | Dataset Size | GPU | Epochs | Time |
|-------|--------------|-----|--------|------|
| [Model] | [#] images | [GPU] | 100 | [#h] |
| | | | | |

*Note: [TO BE FILLED with actual timing data]*
