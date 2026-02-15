#!/usr/bin/env python3
"""
Evaluation script for object detection models.

This script provides comprehensive model evaluation including:
- mAP (mean Average Precision) calculation
- Precision, Recall, F1 scores
- Confusion matrix generation
- Per-class performance analysis
- Prediction visualization
- Evaluation report export
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Base class for model evaluation."""

    def __init__(
        self,
        model_path: Path,
        data_path: Path,
        output_dir: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize model evaluator.

        Args:
            model_path: Path to model weights
            data_path: Path to validation data
            output_dir: Directory for saving results
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS and evaluation
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.results: dict = {}
        self.predictions: list[dict] = []
        self.ground_truths: list[dict] = []

    def calculate_iou(
        self, box1: np.ndarray, box2: np.ndarray, format: str = "xyxy"
    ) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.

        Args:
            box1: First bounding box [x1, y1, x2, y2] or [x, y, w, h]
            box2: Second bounding box
            format: Box format ('xyxy' or 'xywh')

        Returns:
            IoU score
        """
        if format == "xywh":
            # Convert to xyxy format
            box1 = np.array([box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]])
            box2 = np.array([box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]])

        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def calculate_map(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
        iou_threshold: float = 0.5,
    ) -> dict:
        """
        Calculate mean Average Precision (mAP).

        Args:
            predictions: List of predictions for each image
            ground_truths: List of ground truth annotations
            iou_threshold: IoU threshold for considering a prediction correct

        Returns:
            Dictionary containing mAP and per-class AP
        """
        logger.info(f"Calculating mAP @ IoU={iou_threshold}")

        # Organize predictions and ground truths by class
        class_predictions: dict[int, list[dict]] = {}
        class_ground_truths: dict[int, list[dict]] = {}

        for pred in predictions:
            cls = pred["class"]
            if cls not in class_predictions:
                class_predictions[cls] = []
            class_predictions[cls].append(pred)

        for gt in ground_truths:
            cls = gt["class"]
            if cls not in class_ground_truths:
                class_ground_truths[cls] = []
            class_ground_truths[cls].append(gt)

        # Calculate AP for each class
        class_aps = {}
        all_classes = set(
            list(class_predictions.keys()) + list(class_ground_truths.keys())
        )

        for cls in all_classes:
            preds = class_predictions.get(cls, [])
            gts = class_ground_truths.get(cls, [])

            if len(gts) == 0:
                logger.warning(f"No ground truth for class {cls}, skipping")
                continue

            # Sort predictions by confidence (descending)
            preds = sorted(preds, key=lambda x: x["confidence"], reverse=True)

            # Track which ground truths have been matched
            gt_matched = [False] * len(gts)
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))

            for i, pred in enumerate(preds):
                # Find best matching ground truth
                best_iou: float = 0.0
                best_gt_idx = -1

                for j, gt in enumerate(gts):
                    if gt["image_id"] != pred["image_id"]:
                        continue
                    if gt_matched[j]:
                        continue

                    iou = self.calculate_iou(pred["bbox"], gt["bbox"], format="xyxy")

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                # Check if prediction matches ground truth
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tp[i] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[i] = 1

            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / len(gts)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

            # Calculate AP using 11-point interpolation
            ap: float = 0.0
            for t in np.linspace(0, 1, 11):
                if np.sum(recalls >= t) == 0:
                    p: float = 0.0
                else:
                    p = float(np.max(precisions[recalls >= t]))
                ap += p / 11

            class_aps[cls] = ap

        # Calculate mAP
        if len(class_aps) > 0:
            map_score: float = float(np.mean(list(class_aps.values())))
        else:
            map_score = 0.0

        return {"mAP": map_score, "class_AP": class_aps}

    def generate_confusion_matrix(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
        class_names: list[str],
        iou_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Generate confusion matrix.

        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            class_names: List of class names
            iou_threshold: IoU threshold for matching

        Returns:
            Confusion matrix (ground_truth x prediction)
        """
        logger.info("Generating confusion matrix")

        num_classes = len(class_names)
        confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

        # Group by image
        img_predictions: dict[str, list[dict]] = {}
        img_ground_truths: dict[str, list[dict]] = {}

        for pred in predictions:
            img_id = pred["image_id"]
            if img_id not in img_predictions:
                img_predictions[img_id] = []
            img_predictions[img_id].append(pred)

        for gt in ground_truths:
            img_id = gt["image_id"]
            if img_id not in img_ground_truths:
                img_ground_truths[img_id] = []
            img_ground_truths[img_id].append(gt)

        # Process each image
        all_images = set(list(img_predictions.keys()) + list(img_ground_truths.keys()))

        for img_id in all_images:
            preds = img_predictions.get(img_id, [])
            gts = img_ground_truths.get(img_id, [])

            gt_matched = [False] * len(gts)

            # Match predictions to ground truths
            for pred in preds:
                best_iou: float = 0.0
                best_gt_idx = -1

                for j, gt in enumerate(gts):
                    if gt_matched[j]:
                        continue

                    iou = self.calculate_iou(pred["bbox"], gt["bbox"], format="xyxy")

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    # Matched prediction
                    gt_class = gts[best_gt_idx]["class"]
                    pred_class = pred["class"]
                    confusion_matrix[gt_class, pred_class] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    # False positive (predicted but no matching ground truth)
                    confusion_matrix[num_classes, pred["class"]] += 1

            # Count false negatives (ground truth without matching prediction)
            for j, matched in enumerate(gt_matched):
                if not matched:
                    gt_class = gts[j]["class"]
                    confusion_matrix[gt_class, num_classes] += 1

        return confusion_matrix

    def plot_confusion_matrix(
        self, confusion_matrix: np.ndarray, class_names: list[str], output_path: Path
    ) -> None:
        """
        Plot and save confusion matrix.

        Args:
            confusion_matrix: Confusion matrix
            class_names: List of class names
            output_path: Path to save plot
        """
        logger.info(f"Plotting confusion matrix to {output_path}")

        # Add "background" label for false positives/negatives
        labels = class_names + ["background"]

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def visualize_predictions(
        self,
        image_path: Path,
        predictions: list[dict],
        ground_truths: list[dict],
        class_names: list[str],
        output_path: Path,
    ) -> None:
        """
        Visualize predictions on image.

        Args:
            image_path: Path to input image
            predictions: List of predictions for this image
            ground_truths: List of ground truths for this image
            class_names: List of class names
            output_path: Path to save visualization
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot ground truths
        ax1.imshow(img)
        ax1.set_title("Ground Truth")
        ax1.axis("off")

        for gt in ground_truths:
            bbox = gt["bbox"]
            cls = gt["class"]
            class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"

            rect = plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="green",
                linewidth=2,
            )
            ax1.add_patch(rect)
            ax1.text(
                bbox[0],
                bbox[1] - 5,
                class_name,
                color="white",
                fontsize=10,
                bbox={"facecolor": "green", "alpha": 0.7},
            )

        # Plot predictions
        ax2.imshow(img)
        ax2.set_title("Predictions")
        ax2.axis("off")

        for pred in predictions:
            bbox = pred["bbox"]
            cls = pred["class"]
            conf = pred["confidence"]
            class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"

            rect = plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax2.add_patch(rect)
            ax2.text(
                bbox[0],
                bbox[1] - 5,
                f"{class_name} {conf:.2f}",
                color="white",
                fontsize=10,
                bbox={"facecolor": "red", "alpha": 0.7},
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


class YOLOv8Evaluator(ModelEvaluator):
    """Evaluator for YOLOv8 models."""

    def __init__(self, *args, **kwargs):
        """Initialize YOLOv8 evaluator."""
        super().__init__(*args, **kwargs)

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            ) from None

        logger.info(f"Loading YOLOv8 model from {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def evaluate(
        self, save_visualizations: bool = True, max_visualizations: int = 20
    ) -> dict:
        """
        Run full evaluation.

        Args:
            save_visualizations: Whether to save prediction visualizations
            max_visualizations: Maximum number of visualizations to save

        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting evaluation")

        # Use built-in YOLO validation
        logger.info("Running YOLO validation")
        val_results = self.model.val(
            data=str(self.data_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
        )

        # Extract metrics
        results: dict[str, Any] = {
            "mAP50": float(val_results.box.map50),
            "mAP50-95": float(val_results.box.map),
            "precision": (
                float(val_results.box.p.mean())
                if hasattr(val_results.box, "p")
                else 0.0
            ),
            "recall": (
                float(val_results.box.r.mean())
                if hasattr(val_results.box, "r")
                else 0.0
            ),
            "f1": (
                float(val_results.box.f1.mean())
                if hasattr(val_results.box, "f1")
                else 0.0
            ),
        }

        # Per-class metrics
        if hasattr(val_results.box, "ap_class_index"):
            class_names = self.model.names
            results["per_class"] = {}

            for i, class_idx in enumerate(val_results.box.ap_class_index):
                class_name = class_names[int(class_idx)]
                results["per_class"][class_name] = {
                    "AP50": float(val_results.box.ap50[i]),
                    "AP": float(val_results.box.ap[i]),
                }

        # Log results
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Results")
        logger.info("=" * 60)
        logger.info(f"mAP@0.5: {results['mAP50']:.4f}")
        logger.info(f"mAP@0.5:0.95: {results['mAP50-95']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1 Score: {results['f1']:.4f}")

        if "per_class" in results:
            logger.info("\nPer-class metrics:")
            for class_name, metrics in results["per_class"].items():
                logger.info(f"  {class_name}:")
                logger.info(f"    AP@0.5: {metrics['AP50']:.4f}")
                logger.info(f"    AP@0.5:0.95: {metrics['AP']:.4f}")

        logger.info("=" * 60 + "\n")

        # Save results
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

        # Generate visualizations if requested
        if save_visualizations:
            self.generate_visualizations(max_visualizations)

        self.results = results
        return results

    def generate_visualizations(self, max_images: int = 20) -> None:
        """
        Generate prediction visualizations.

        Args:
            max_images: Maximum number of images to visualize
        """
        logger.info(f"Generating visualizations (max {max_images} images)")

        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Get validation images
        import yaml

        with open(self.data_path) as f:
            data_config = yaml.safe_load(f)

        val_images_dir = Path(data_config["path"]) / data_config["val"]
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(val_images_dir.glob(ext))

        image_files = sorted(image_files)[:max_images]

        for img_path in tqdm(image_files, desc="Generating visualizations"):
            # Run prediction
            results = self.model.predict(
                str(img_path),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            # Save annotated image
            if len(results) > 0:
                result = results[0]
                output_path = viz_dir / f"{img_path.stem}_pred.jpg"
                result.save(str(output_path))

        logger.info(f"Visualizations saved to {viz_dir}")


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate object detection models")

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model weights",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to data.yaml configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/evaluate"),
        help="Output directory for results (default: runs/evaluate)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable prediction visualizations",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=20,
        help="Maximum number of visualizations (default: 20)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="yolov8",
        choices=["yolov8"],
        help="Model type (default: yolov8)",
    )

    args = parser.parse_args()

    # Create evaluator
    if args.model_type == "yolov8":
        evaluator = YOLOv8Evaluator(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Run evaluation
    evaluator.evaluate(
        save_visualizations=not args.no_visualizations,
        max_visualizations=args.max_visualizations,
    )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
