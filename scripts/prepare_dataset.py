#!/usr/bin/env python3
"""
Dataset preparation and validation script for qontinui-finetune.

This script handles dataset conversion, validation, splitting, and statistics
generation for GUI element detection datasets.

Supports:
- Format conversion: COCO, YOLO, Pascal VOC
- Dataset validation and integrity checks
- Train/val/test splitting
- Dataset statistics and visualization
"""

import argparse
import json
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DatasetConverter:
    """Convert between different annotation formats."""

    def __init__(self, input_path: Path, output_path: Path):
        """
        Initialize the dataset converter.

        Args:
            input_path: Path to input dataset
            output_path: Path to output dataset
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def coco_to_yolo(
        self, coco_json: Path, class_names: Optional[List[str]] = None
    ) -> None:
        """
        Convert COCO format to YOLO format.

        Args:
            coco_json: Path to COCO JSON file
            class_names: Optional list of class names (extracted from COCO if None)
        """
        logger.info(f"Converting COCO to YOLO: {coco_json}")

        with open(coco_json, "r") as f:
            coco_data = json.load(f)

        # Extract class names if not provided
        if class_names is None:
            class_names = [cat["name"] for cat in coco_data["categories"]]

        # Create class mapping
        cat_id_to_idx = {
            cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])
        }

        # Create images directory
        labels_dir = self.output_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Group annotations by image
        img_to_anns = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # Convert each image's annotations
        for img in tqdm(coco_data["images"], desc="Converting annotations"):
            img_id = img["id"]
            img_width = img["width"]
            img_height = img["height"]
            img_filename = Path(img["file_name"])

            # Create label file
            label_file = labels_dir / f"{img_filename.stem}.txt"

            if img_id not in img_to_anns:
                # Create empty label file if no annotations
                label_file.touch()
                continue

            with open(label_file, "w") as f:
                for ann in img_to_anns[img_id]:
                    # COCO bbox format: [x, y, width, height]
                    bbox = ann["bbox"]
                    x, y, w, h = bbox

                    # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    norm_width = w / img_width
                    norm_height = h / img_height

                    class_idx = cat_id_to_idx[ann["category_id"]]

                    f.write(
                        f"{class_idx} {x_center:.6f} {y_center:.6f} "
                        f"{norm_width:.6f} {norm_height:.6f}\n"
                    )

        # Save class names
        with open(self.output_path / "classes.txt", "w") as f:
            f.write("\n".join(class_names))

        logger.info(f"Conversion complete. Labels saved to {labels_dir}")

    def yolo_to_coco(
        self, labels_dir: Path, images_dir: Path, class_names: List[str]
    ) -> None:
        """
        Convert YOLO format to COCO format.

        Args:
            labels_dir: Directory containing YOLO label files
            images_dir: Directory containing images
            class_names: List of class names
        """
        logger.info("Converting YOLO to COCO format")

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": idx, "name": name, "supercategory": "gui_element"}
                for idx, name in enumerate(class_names)
            ],
        }

        ann_id = 0
        label_files = sorted(Path(labels_dir).glob("*.txt"))

        for img_id, label_file in enumerate(tqdm(label_files, desc="Converting")):
            # Find corresponding image
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = images_dir / f"{label_file.stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                logger.warning(f"No image found for {label_file}")
                continue

            # Get image dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size

            # Add image info
            coco_data["images"].append(
                {
                    "id": img_id,
                    "file_name": img_path.name,
                    "width": img_width,
                    "height": img_height,
                }
            )

            # Convert annotations
            with open(label_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    parts = line.strip().split()
                    class_idx = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # Convert from YOLO to COCO format
                    x = (x_center - width / 2) * img_width
                    y = (y_center - height / 2) * img_height
                    w = width * img_width
                    h = height * img_height

                    coco_data["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_idx,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

        # Save COCO JSON
        output_file = self.output_path / "annotations.json"
        with open(output_file, "w") as f:
            json.dump(coco_data, f, indent=2)

        logger.info(f"Conversion complete. Saved to {output_file}")

    def pascal_voc_to_yolo(self, annotations_dir: Path, images_dir: Path) -> None:
        """
        Convert Pascal VOC format to YOLO format.

        Args:
            annotations_dir: Directory containing XML annotation files
            images_dir: Directory containing images
        """
        logger.info("Converting Pascal VOC to YOLO format")

        # Extract class names
        class_names = set()
        for xml_file in Path(annotations_dir).glob("*.xml"):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall("object"):
                class_names.add(obj.find("name").text)

        class_names = sorted(class_names)
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Create labels directory
        labels_dir = self.output_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Convert each XML file
        xml_files = sorted(Path(annotations_dir).glob("*.xml"))
        for xml_file in tqdm(xml_files, desc="Converting annotations"):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get image dimensions
            size = root.find("size")
            img_width = int(size.find("width").text)
            img_height = int(size.find("height").text)

            # Create label file
            label_file = labels_dir / f"{xml_file.stem}.txt"

            with open(label_file, "w") as f:
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    class_idx = class_to_idx[class_name]

                    bbox = obj.find("bndbox")
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)

                    # Convert to YOLO format
                    x_center = ((xmin + xmax) / 2) / img_width
                    y_center = ((ymin + ymax) / 2) / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height

                    f.write(
                        f"{class_idx} {x_center:.6f} {y_center:.6f} "
                        f"{width:.6f} {height:.6f}\n"
                    )

        # Save class names
        with open(self.output_path / "classes.txt", "w") as f:
            f.write("\n".join(class_names))

        logger.info(f"Conversion complete. Labels saved to {labels_dir}")


class DatasetValidator:
    """Validate dataset structure and integrity."""

    def __init__(self, dataset_path: Path):
        """
        Initialize the dataset validator.

        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)

    def validate_yolo_dataset(self) -> Tuple[bool, List[str]]:
        """
        Validate YOLO format dataset.

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        logger.info(f"Validating YOLO dataset at {self.dataset_path}")

        # Check for required structure
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"

        if not images_dir.exists():
            errors.append(f"Images directory not found: {images_dir}")

        if not labels_dir.exists():
            errors.append(f"Labels directory not found: {labels_dir}")

        if errors:
            return False, errors

        # Check image-label pairs
        image_files = set()
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.update(
                f.stem for f in images_dir.glob(f"*{ext}") if f.is_file()
            )

        label_files = set(f.stem for f in labels_dir.glob("*.txt") if f.is_file())

        # Find missing labels
        missing_labels = image_files - label_files
        if missing_labels:
            errors.append(
                f"Found {len(missing_labels)} images without labels: "
                f"{list(missing_labels)[:5]}..."
            )

        # Find orphaned labels
        orphaned_labels = label_files - image_files
        if orphaned_labels:
            errors.append(
                f"Found {len(orphaned_labels)} labels without images: "
                f"{list(orphaned_labels)[:5]}..."
            )

        # Validate label files
        for label_file in tqdm(
            list(labels_dir.glob("*.txt")), desc="Validating labels"
        ):
            try:
                with open(label_file, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue

                        parts = line.strip().split()
                        if len(parts) < 5:
                            errors.append(
                                f"{label_file.name}:{line_num} - "
                                f"Invalid format (expected 5 values, got {len(parts)})"
                            )
                            continue

                        # Validate coordinates are in [0, 1] range
                        _, x, y, w, h = map(float, parts[:5])
                        if not (
                            0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1
                        ):
                            errors.append(
                                f"{label_file.name}:{line_num} - "
                                f"Coordinates out of range [0, 1]: {x}, {y}, {w}, {h}"
                            )

            except Exception as e:
                errors.append(f"Error reading {label_file.name}: {e}")

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Dataset validation successful!")
        else:
            logger.error(f"Dataset validation failed with {len(errors)} errors")

        return is_valid, errors


class DatasetSplitter:
    """Split dataset into train/val/test sets."""

    def __init__(self, dataset_path: Path, output_path: Path):
        """
        Initialize the dataset splitter.

        Args:
            dataset_path: Path to source dataset
            output_path: Path to output split dataset
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)

    def split_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            seed: Random seed for reproducibility
        """
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        logger.info(
            f"Splitting dataset: train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

        np.random.seed(seed)

        # Get all image files
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"

        image_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.extend(images_dir.glob(f"*{ext}"))

        image_files = sorted(image_files)
        logger.info(f"Found {len(image_files)} images")

        # Shuffle and split
        indices = np.random.permutation(len(image_files))
        train_end = int(len(indices) * train_ratio)
        val_end = train_end + int(len(indices) * val_ratio)

        splits = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }

        # Create split directories
        for split_name, split_indices in splits.items():
            logger.info(f"Creating {split_name} split with {len(split_indices)} images")

            split_images_dir = self.output_path / split_name / "images"
            split_labels_dir = self.output_path / split_name / "labels"
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)

            for idx in tqdm(split_indices, desc=f"Copying {split_name}"):
                img_file = image_files[idx]
                label_file = labels_dir / f"{img_file.stem}.txt"

                # Copy image
                shutil.copy2(img_file, split_images_dir / img_file.name)

                # Copy label if exists
                if label_file.exists():
                    shutil.copy2(label_file, split_labels_dir / label_file.name)

        # Create YOLO data.yaml
        class_names = []
        classes_file = self.dataset_path / "classes.txt"
        if classes_file.exists():
            with open(classes_file, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]

        data_yaml = {
            "path": str(self.output_path.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names,
        }

        with open(self.output_path / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        logger.info(f"Split complete. Dataset saved to {self.output_path}")


class DatasetStatistics:
    """Generate dataset statistics and visualizations."""

    def __init__(self, dataset_path: Path):
        """
        Initialize dataset statistics generator.

        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)

    def generate_statistics(self) -> Dict:
        """
        Generate comprehensive dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        logger.info("Generating dataset statistics")

        labels_dir = self.dataset_path / "labels"
        images_dir = self.dataset_path / "images"

        stats = {
            "num_images": 0,
            "num_annotations": 0,
            "class_distribution": {},
            "bbox_sizes": [],
            "image_sizes": [],
            "annotations_per_image": [],
        }

        # Load class names
        classes_file = self.dataset_path / "classes.txt"
        if classes_file.exists():
            with open(classes_file, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]
            stats["class_names"] = class_names
        else:
            class_names = []

        # Analyze labels
        label_files = sorted(labels_dir.glob("*.txt"))
        stats["num_images"] = len(label_files)

        for label_file in tqdm(label_files, desc="Analyzing dataset"):
            anns_in_image = 0

            with open(label_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    parts = line.strip().split()
                    class_idx = int(parts[0])
                    _, _, w, h = map(float, parts[1:5])

                    # Update class distribution
                    if class_idx not in stats["class_distribution"]:
                        stats["class_distribution"][class_idx] = 0
                    stats["class_distribution"][class_idx] += 1

                    # Store bbox size
                    stats["bbox_sizes"].append((w, h))

                    anns_in_image += 1
                    stats["num_annotations"] += 1

            stats["annotations_per_image"].append(anns_in_image)

            # Get image size
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = images_dir / f"{label_file.stem}{ext}"
                if img_path.exists():
                    with Image.open(img_path) as img:
                        stats["image_sizes"].append(img.size)
                    break

        # Calculate summary statistics
        if stats["bbox_sizes"]:
            widths, heights = zip(*stats["bbox_sizes"])
            stats["bbox_stats"] = {
                "mean_width": float(np.mean(widths)),
                "mean_height": float(np.mean(heights)),
                "median_width": float(np.median(widths)),
                "median_height": float(np.median(heights)),
            }

        if stats["annotations_per_image"]:
            stats["annotations_stats"] = {
                "mean": float(np.mean(stats["annotations_per_image"])),
                "median": float(np.median(stats["annotations_per_image"])),
                "max": int(np.max(stats["annotations_per_image"])),
                "min": int(np.min(stats["annotations_per_image"])),
            }

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("Dataset Statistics Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total Images: {stats['num_images']}")
        logger.info(f"Total Annotations: {stats['num_annotations']}")
        logger.info("\nClass Distribution:")
        for class_idx, count in sorted(stats["class_distribution"].items()):
            class_name = (
                class_names[class_idx]
                if class_idx < len(class_names)
                else f"Class {class_idx}"
            )
            logger.info(f"  {class_name}: {count}")
        logger.info("\nAnnotations per Image:")
        logger.info(f"  Mean: {stats['annotations_stats']['mean']:.2f}")
        logger.info(f"  Median: {stats['annotations_stats']['median']:.2f}")
        logger.info(f"  Max: {stats['annotations_stats']['max']}")
        logger.info(f"  Min: {stats['annotations_stats']['min']}")
        logger.info(f"{'='*60}\n")

        # Save statistics to JSON
        output_file = self.dataset_path / "statistics.json"
        # Convert numpy types to native Python types for JSON serialization
        stats_json = {
            k: (
                v
                if not isinstance(v, (np.integer, np.floating, np.ndarray))
                else v.tolist() if isinstance(v, np.ndarray) else float(v)
            )
            for k, v in stats.items()
            if k not in ["bbox_sizes", "image_sizes", "annotations_per_image"]
        }

        with open(output_file, "w") as f:
            json.dump(stats_json, f, indent=2)

        logger.info(f"Statistics saved to {output_file}")

        return stats


def main():
    """Main entry point for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare and validate datasets for GUI element detection"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert dataset format")
    convert_parser.add_argument(
        "--input", type=Path, required=True, help="Input dataset path"
    )
    convert_parser.add_argument(
        "--output", type=Path, required=True, help="Output dataset path"
    )
    convert_parser.add_argument(
        "--from-format",
        choices=["coco", "yolo", "voc"],
        required=True,
        help="Source format",
    )
    convert_parser.add_argument(
        "--to-format",
        choices=["coco", "yolo", "voc"],
        required=True,
        help="Target format",
    )
    convert_parser.add_argument(
        "--coco-json", type=Path, help="COCO JSON file (for COCO format)"
    )
    convert_parser.add_argument("--images-dir", type=Path, help="Images directory")
    convert_parser.add_argument(
        "--labels-dir", type=Path, help="Labels directory (for YOLO format)"
    )
    convert_parser.add_argument(
        "--annotations-dir", type=Path, help="Annotations directory (for VOC format)"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument(
        "--dataset", type=Path, required=True, help="Dataset path"
    )
    validate_parser.add_argument(
        "--format", choices=["yolo"], default="yolo", help="Dataset format"
    )

    # Split command
    split_parser = subparsers.add_parser("split", help="Split dataset")
    split_parser.add_argument(
        "--dataset", type=Path, required=True, help="Input dataset path"
    )
    split_parser.add_argument(
        "--output", type=Path, required=True, help="Output path for split dataset"
    )
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    split_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)",
    )
    split_parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)"
    )
    split_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # Statistics command
    stats_parser = subparsers.add_parser("stats", help="Generate dataset statistics")
    stats_parser.add_argument(
        "--dataset", type=Path, required=True, help="Dataset path"
    )

    args = parser.parse_args()

    if args.command == "convert":
        converter = DatasetConverter(args.input, args.output)

        if args.from_format == "coco" and args.to_format == "yolo":
            if not args.coco_json:
                parser.error("--coco-json required for COCO to YOLO conversion")
            converter.coco_to_yolo(args.coco_json)

        elif args.from_format == "yolo" and args.to_format == "coco":
            if not args.labels_dir or not args.images_dir:
                parser.error(
                    "--labels-dir and --images-dir required for YOLO to COCO conversion"
                )
            # Need class names
            classes_file = args.input / "classes.txt"
            if not classes_file.exists():
                parser.error(f"classes.txt not found in {args.input}")
            with open(classes_file, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]
            converter.yolo_to_coco(args.labels_dir, args.images_dir, class_names)

        elif args.from_format == "voc" and args.to_format == "yolo":
            if not args.annotations_dir or not args.images_dir:
                parser.error(
                    "--annotations-dir and --images-dir required for VOC to YOLO conversion"
                )
            converter.pascal_voc_to_yolo(args.annotations_dir, args.images_dir)

        else:
            parser.error(
                f"Conversion from {args.from_format} to {args.to_format} not implemented"
            )

    elif args.command == "validate":
        validator = DatasetValidator(args.dataset)
        is_valid, errors = validator.validate_yolo_dataset()

        if not is_valid:
            logger.error("Validation errors:")
            for error in errors[:10]:  # Show first 10 errors
                logger.error(f"  - {error}")
            if len(errors) > 10:
                logger.error(f"  ... and {len(errors) - 10} more errors")
            exit(1)

    elif args.command == "split":
        splitter = DatasetSplitter(args.dataset, args.output)
        splitter.split_dataset(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    elif args.command == "stats":
        stats_gen = DatasetStatistics(args.dataset)
        stats_gen.generate_statistics()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
