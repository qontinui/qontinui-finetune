# Deployment Guide: Export and Deployment

## Overview

This guide covers exporting trained models, preparing them for deployment, and deploying them in various environments (cloud, edge, mobile, etc.).

---

## 1. Model Export

### 1.1 Export Preparation

**Pre-export checklist:**

- [ ] Model training completed
- [ ] Best checkpoint identified at: [Path]
- [ ] Validation metrics recorded
- [ ] Model performance acceptable: [Metric] >= [Target]
- [ ] Export configuration prepared

### 1.2 Model-Specific Export

#### YOLO v8 Export

**Export script location:** `/home/user/qontinui-finetune/scripts/export_yolo.py` [TO BE FILLED]

**Available export formats:**

| Format | Extension | Framework | Use Case | Size (approx.) |
|--------|-----------|-----------|----------|----------------|
| PyTorch | .pt | PyTorch | Development, Training | [Size] |
| ONNX | .onnx | ONNX Runtime | Cross-platform inference | [Size] |
| TensorFlow | .pb | TensorFlow | TF-based inference | [Size] |
| TensorFlow Lite | .tflite | TFLite | Mobile/Edge | [Size] |
| CoreML | .mlmodel | CoreML | iOS/macOS | [Size] |
| NCNN | .param/.bin | NCNN | Mobile optimization | [Size] |

**Export to PyTorch:**
```bash
python /home/user/qontinui-finetune/scripts/export_yolo.py \
  --model-path /home/user/qontinui-finetune/outputs/run_1/checkpoints/best_model.pt \
  --format pytorch \
  --output-dir /home/user/qontinui-finetune/exports/yolo_v8
```

**Export to ONNX:**
```bash
python /home/user/qontinui-finetune/scripts/export_yolo.py \
  --model-path /home/user/qontinui-finetune/outputs/run_1/checkpoints/best_model.pt \
  --format onnx \
  --opset-version 13 \
  --output-dir /home/user/qontinui-finetune/exports/yolo_v8_onnx \
  --optimize
```

**Export to TensorFlow Lite:**
```bash
python /home/user/qontinui-finetune/scripts/export_yolo.py \
  --model-path /home/user/qontinui-finetune/outputs/run_1/checkpoints/best_model.pt \
  --format tflite \
  --output-dir /home/user/qontinui-finetune/exports/yolo_v8_lite \
  --quantize int8 \
  --representative-data /home/user/qontinui-finetune/data/processed/train
```

#### Detectron2 Export

**Export script location:** `/home/user/qontinui-finetune/scripts/export_detectron2.py` [TO BE FILLED]

**Supported formats:**

```bash
# Export to TorchScript
python /home/user/qontinui-finetune/scripts/export_detectron2.py \
  --model-path /home/user/qontinui-finetune/outputs/run_1/checkpoints/best_model.pt \
  --format torchscript \
  --output-dir /home/user/qontinui-finetune/exports/detectron2_torchscript

# Export to ONNX
python /home/user/qontinui-finetune/scripts/export_detectron2.py \
  --model-path /home/user/qontinui-finetune/outputs/run_1/checkpoints/best_model.pt \
  --format onnx \
  --output-dir /home/user/qontinui-finetune/exports/detectron2_onnx
```

#### SAM Export

**Export script location:** `/home/user/qontinui-finetune/scripts/export_sam.py` [TO BE FILLED]

```bash
python /home/user/qontinui-finetune/scripts/export_sam.py \
  --model-path /home/user/qontinui-finetune/outputs/run_1/checkpoints/best_model.pt \
  --format onnx \
  --optimize \
  --output-dir /home/user/qontinui-finetune/exports/sam_onnx
```

### 1.3 Quantization

**Model quantization for deployment:**

**Quantization options:**

| Type | Size Reduction | Speed Improvement | Accuracy Loss | Hardware |
|------|---------------|------------------|--------------|----------|
| INT8 | 4x | 3-4x | ~1-2% | CPU, Mobile |
| INT4 | 8x | 4-6x | ~2-5% | Mobile, Edge |
| FP16 | 2x | 1.5-2x | Minimal | GPU (with Tensor Cores) |
| INT8 + FP16 | Variable | Variable | Varies | GPU + CPU |

**Post-training quantization:**
```bash
python /home/user/qontinui-finetune/scripts/quantize_model.py \
  --model-path /home/user/qontinui-finetune/exports/yolo_v8_onnx/model.onnx \
  --quantization-type int8 \
  --calibration-data /home/user/qontinui-finetune/data/processed/train \
  --output-path /home/user/qontinui-finetune/exports/yolo_v8_quantized/model.onnx
```

### 1.4 Export Verification

**Verify exported model:**
```bash
python /home/user/qontinui-finetune/scripts/verify_export.py \
  --model-path /home/user/qontinui-finetune/exports/yolo_v8_onnx/model.onnx \
  --format onnx \
  --test-image /home/user/qontinui-finetune/data/processed/test/sample.jpg \
  --compare-with /home/user/qontinui-finetune/outputs/run_1/checkpoints/best_model.pt
```

---

## 2. Model Optimization

### 2.1 Optimization Techniques

**Available optimization methods:**

| Technique | Complexity | Speed Gain | Memory Reduction | Notes |
|-----------|-----------|-----------|------------------|-------|
| Quantization | Medium | 2-6x | 2-8x | Most impactful for inference |
| Pruning | High | 1-2x | 1-2x | Requires retraining |
| Distillation | High | Variable | Variable | Requires teacher model |
| Graph optimization | Low | 1-1.5x | Minimal | Usually included in exporters |
| Operator fusion | Low | 1-1.5x | Minimal | Automatic in most frameworks |

### 2.2 Graph Optimization

**Optimize ONNX model graph:**
```bash
python /home/user/qontinui-finetune/scripts/optimize_onnx.py \
  --model-path /home/user/qontinui-finetune/exports/yolo_v8_onnx/model.onnx \
  --optimization-level all \
  --output-path /home/user/qontinui-finetune/exports/yolo_v8_onnx_optimized/model.onnx
```

### 2.3 Benchmarking

**Benchmark model performance:**

```bash
python /home/user/qontinui-finetune/scripts/benchmark.py \
  --model-path /home/user/qontinui-finetune/exports/yolo_v8_onnx/model.onnx \
  --format onnx \
  --input-shape 1 3 640 640 \
  --num-runs 100 \
  --warmup-runs 10 \
  --output-report /home/user/qontinui-finetune/exports/benchmark_results.json
```

**Benchmark comparison:**

| Model | Format | Quantization | Latency (ms) | Memory (MB) | Throughput (fps) |
|-------|--------|--------------|--------------|-------------|------------------|
| YOLO v8 | PyTorch | - | [Value] | [Value] | [Value] |
| YOLO v8 | ONNX | - | [Value] | [Value] | [Value] |
| YOLO v8 | ONNX | INT8 | [Value] | [Value] | [Value] |
| | | | | | |

---

## 3. Deployment Scenarios

### 3.1 Python/PyTorch Deployment

**Inference code template:**

```python
import torch
from pathlib import Path

# Load model
model_path = Path("/home/user/qontinui-finetune/exports/yolo_v8/model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(str(model_path))
model = model.to(device)
model.eval()

# Inference
with torch.no_grad():
    image = torch.randn(1, 3, 640, 640).to(device)
    output = model(image)

# Post-process results
predictions = [TO BE FILLED: post-processing logic]
```

**Deployment package structure:**
```
deployment/
├── models/
│   ├── best_model.pt
│   └── model_config.yaml
├── inference.py
├── postprocess.py
├── requirements.txt
└── README.md
```

### 3.2 ONNX Runtime Deployment

**ONNX inference code:**

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
model_path = "/home/user/qontinui-finetune/exports/yolo_v8_onnx/model.onnx"
sess = ort.InferenceSession(model_path)

# Get input/output names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Prepare input
image = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Run inference
outputs = sess.run([output_name], {input_name: image})
predictions = outputs[0]
```

### 3.3 Docker Containerization

**Dockerfile template:**

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Copy model and code
COPY exports/yolo_v8_onnx/ /app/models/
COPY inference.py /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "-m", "uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
# Build image
docker build -t qontinui-inference:latest .

# Run container
docker run --gpus all -p 8000:8000 qontinui-inference:latest
```

### 3.4 Cloud Deployment

#### AWS SageMaker Deployment

**Model serving container:**
```bash
# Create SageMaker-compatible container
python /home/user/qontinui-finetune/scripts/prepare_sagemaker.py \
  --model-path /home/user/qontinui-finetune/exports/yolo_v8_onnx/model.onnx \
  --output-dir /home/user/qontinui-finetune/exports/sagemaker_model
```

#### Google Cloud AI Platform

**Prepare model for AI Platform:**
```bash
python /home/user/qontinui-finetune/scripts/prepare_gcp.py \
  --model-path /home/user/qontinui-finetune/exports/yolo_v8_onnx/model.onnx \
  --output-dir /home/user/qontinui-finetune/exports/gcp_model
```

### 3.5 Edge Deployment

#### TensorFlow Lite (Mobile/Edge)

**Deploy TFLite model:**

```bash
# Convert to TFLite
python /home/user/qontinui-finetune/scripts/export_yolo.py \
  --model-path /home/user/qontinui-finetune/outputs/run_1/checkpoints/best_model.pt \
  --format tflite \
  --quantize int8 \
  --output-dir /home/user/qontinui-finetune/exports/tflite_model

# Copy to mobile project
cp /home/user/qontinui-finetune/exports/tflite_model/model.tflite \
   /path/to/mobile/project/assets/
```

#### NCNN (Mobile)

**Export to NCNN format:**
```bash
python /home/user/qontinui-finetune/scripts/export_yolo.py \
  --model-path /home/user/qontinui-finetune/outputs/run_1/checkpoints/best_model.pt \
  --format ncnn \
  --output-dir /home/user/qontinui-finetune/exports/ncnn_model
```

### 3.6 Real-time Inference Server

**FastAPI inference server:**

```python
from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
from PIL import Image
import numpy as np

app = FastAPI()

# Load model
model = ort.InferenceSession(
    "/home/user/qontinui-finetune/exports/yolo_v8_onnx/model.onnx"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(file.file)

    # Preprocess
    image_array = np.array(image).astype(np.float32) / 255.0
    # [Additional preprocessing as needed]

    # Inference
    input_name = model.get_inputs()[0].name
    output = model.run(None, {input_name: image_array})

    # Post-process
    predictions = [TO BE FILLED: post-processing]

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Start server:**
```bash
python inference_server.py
```

---

## 4. Performance Monitoring

### 4.1 Metrics to Monitor

**In-production metrics:**

| Metric | Description | Target | Alert Threshold |
|--------|-------------|--------|-----------------|
| Inference latency | Time to process one image | [#ms] | > [#ms] |
| Throughput | Images per second | [#fps] | < [#fps] |
| Memory usage | Peak memory during inference | [#MB] | > [#MB] |
| GPU utilization | % GPU usage | [#%] | < [#%] |
| Error rate | % failed inferences | < 1% | > 5% |
| Model accuracy | Online accuracy tracking | [#%] | < [#%] |

### 4.2 Logging and Monitoring

**Setup monitoring:**

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="/home/user/qontinui-finetune/logs/inference.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def inference_with_monitoring(image):
    start_time = datetime.now()

    # Run inference
    output = model.predict(image)

    latency = (datetime.now() - start_time).total_seconds() * 1000
    logger.info(f"Inference completed in {latency:.2f}ms")

    return output
```

### 4.3 Drift Detection

**Monitor for model drift:**

```bash
python /home/user/qontinui-finetune/scripts/detect_drift.py \
  --baseline-metrics /home/user/qontinui-finetune/outputs/run_1/evaluation/metrics.json \
  --current-metrics /path/to/current/metrics.json \
  --threshold 0.05 \
  --output-report /home/user/qontinui-finetune/reports/drift_report.json
```

---

## 5. Version Control and Rollback

### 5.1 Model Registry

**Model versioning structure:**

```
exports/
├── yolo_v8/
│   ├── v1.0/
│   │   ├── model.pt
│   │   ├── metadata.json
│   │   └── metrics.json
│   ├── v1.1/
│   │   ├── model.pt
│   │   ├── metadata.json
│   │   └── metrics.json
│   └── latest -> v1.1/
```

**Metadata file template:**

```json
{
  "model_id": "yolo_v8_v1.0",
  "created_date": "2024-MM-DD",
  "training_dataset": "game-ui + gui-elements",
  "metrics": {
    "val_mAP": 0.XXX,
    "test_mAP": 0.XXX
  },
  "input_shape": [640, 640, 3],
  "output_format": "predictions with confidence",
  "deployment_status": "production"
}
```

### 5.2 Rollback Procedure

**Rollback to previous model version:**

```bash
# List available versions
ls -la /home/user/qontinui-finetune/exports/yolo_v8/

# Rollback to previous version
cp /home/user/qontinui-finetune/exports/yolo_v8/v1.0/model.pt \
   /home/user/qontinui-finetune/exports/yolo_v8/latest/model.pt

# Update deployment
docker pull qontinui-inference:v1.0
docker run --gpus all -p 8000:8000 qontinui-inference:v1.0
```

---

## 6. Troubleshooting Deployment

### 6.1 Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Model too large | OOM errors, slow loading | Use quantization, pruning |
| Poor inference accuracy | Predictions incorrect in production | Check preprocessing, model version |
| Latency too high | Inference slower than expected | Profile code, use optimization |
| Memory leaks | Increasing memory usage over time | Check resource cleanup |
| Model drift | Performance degradation over time | Monitor metrics, retrain |

### 6.2 Performance Debugging

**Profile inference:**
```bash
python /home/user/qontinui-finetune/scripts/profile_inference.py \
  --model-path /home/user/qontinui-finetune/exports/yolo_v8_onnx/model.onnx \
  --input-shape 1 3 640 640 \
  --num-iterations 100 \
  --output-report /home/user/qontinui-finetune/reports/profile.json
```

---

## 7. Deployment Checklist

### Pre-deployment

- [ ] Model exported and verified
- [ ] Quantization applied (if needed)
- [ ] Inference latency acceptable: [#ms]
- [ ] Model accuracy verified on test set
- [ ] All dependencies documented
- [ ] Docker image built and tested
- [ ] Documentation updated

### Deployment

- [ ] Staging environment tested
- [ ] Monitoring setup complete
- [ ] Rollback procedure tested
- [ ] Team trained on new model
- [ ] Deployment completed

### Post-deployment

- [ ] Monitor inference metrics
- [ ] Track model accuracy
- [ ] Document any issues
- [ ] Plan retraining schedule
- [ ] Collect feedback for improvements

---

## 8. Related Documentation

- `/home/user/qontinui-finetune/docs/model-selection.md`
- `/home/user/qontinui-finetune/docs/training-guide.md`
- `/home/user/qontinui-finetune/docs/dataset-creation.md`
- `/home/user/qontinui-finetune/README.md`

---

## Appendix: Quick Reference

### Export Command Reference

```bash
# YOLO v8 to ONNX
python scripts/export_yolo.py \
  --model-path outputs/run_1/checkpoints/best_model.pt \
  --format onnx \
  --output-dir exports/yolo_v8_onnx

# Quantize ONNX
python scripts/quantize_model.py \
  --model-path exports/yolo_v8_onnx/model.onnx \
  --quantization-type int8 \
  --output-path exports/yolo_v8_quantized/model.onnx

# Benchmark
python scripts/benchmark.py \
  --model-path exports/yolo_v8_onnx/model.onnx \
  --format onnx

# Start inference server
python inference_server.py
```

### Model Size Reference

| Model | Original Size | Quantized (INT8) | Quantized (INT4) |
|-------|---------------|------------------|------------------|
| YOLO v8-nano | [#MB] | [#MB] | [#MB] |
| YOLO v8-small | [#MB] | [#MB] | [#MB] |
| YOLO v8-medium | [#MB] | [#MB] | [#MB] |

*Note: [TO BE FILLED with actual data]*
