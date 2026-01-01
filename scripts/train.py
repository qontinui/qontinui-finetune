#!/usr/bin/env python3
"""
Training script for fine-tuning object detection models.

This script handles fine-tuning of pre-trained models (YOLOv8, Detectron2, etc.)
for GUI element detection with support for:
- Multiple model architectures
- Configurable hyperparameters
- Training monitoring and logging
- Model checkpointing
- Resume training from checkpoints
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from qontinui_schemas.common import utc_now

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class YOLOv8Trainer:
    """Trainer for YOLOv8 models."""

    def __init__(
        self,
        model_size: str = "n",
        data_config: Path | None = None,
        project: Path | None = None,
        name: str = "train",
        exist_ok: bool = False,
    ):
        """
        Initialize YOLOv8 trainer.

        Args:
            model_size: Model size (n, s, m, l, x)
            data_config: Path to data.yaml configuration
            project: Project directory for saving results
            name: Experiment name
            exist_ok: Whether to overwrite existing results
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            ) from None

        self.model_size = model_size
        self.data_config = Path(data_config) if data_config else None
        self.project = Path(project) if project else Path("runs/train")
        self.name = name
        self.exist_ok = exist_ok

        # Initialize model
        model_path = f"yolov8{model_size}.pt"
        logger.info(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: int = 3,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        augment: bool = True,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        flipud: float = 0.0,
        fliplr: float = 0.5,
        mosaic: float = 1.0,
        mixup: float = 0.0,
        copy_paste: float = 0.0,
        patience: int = 50,
        save_period: int = -1,
        workers: int = 8,
        device: str | None = None,
        resume: bool = False,
        amp: bool = True,
        fraction: float = 1.0,
        **kwargs,
    ) -> dict:
        """
        Train the YOLOv8 model.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            imgsz: Input image size
            lr0: Initial learning rate
            lrf: Final learning rate factor
            momentum: SGD momentum
            weight_decay: Weight decay
            warmup_epochs: Warmup epochs
            warmup_momentum: Warmup momentum
            warmup_bias_lr: Warmup bias learning rate
            augment: Whether to use augmentation
            hsv_h: HSV-Hue augmentation
            hsv_s: HSV-Saturation augmentation
            hsv_v: HSV-Value augmentation
            degrees: Rotation augmentation (degrees)
            translate: Translation augmentation (fraction)
            scale: Scale augmentation (gain)
            shear: Shear augmentation (degrees)
            perspective: Perspective augmentation (probability)
            flipud: Vertical flip augmentation (probability)
            fliplr: Horizontal flip augmentation (probability)
            mosaic: Mosaic augmentation (probability)
            mixup: MixUp augmentation (probability)
            copy_paste: Copy-paste augmentation (probability)
            patience: Early stopping patience (epochs)
            save_period: Save checkpoint every n epochs (-1 to disable)
            workers: Number of dataloader workers
            device: Device to train on (cuda, cpu, or device id)
            resume: Resume training from last checkpoint
            amp: Use automatic mixed precision
            fraction: Fraction of dataset to use
            **kwargs: Additional training arguments

        Returns:
            Dictionary containing training results
        """
        if self.data_config is None:
            raise ValueError("data_config must be provided for training")

        if not self.data_config.exists():
            raise FileNotFoundError(f"Data config not found: {self.data_config}")

        logger.info("Starting training with configuration:")
        logger.info(f"  Model: YOLOv8{self.model_size}")
        logger.info(f"  Data: {self.data_config}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Image size: {imgsz}")
        logger.info(f"  Learning rate: {lr0}")
        logger.info(f"  Device: {device or 'auto'}")

        # Prepare training arguments
        train_args = {
            "data": str(self.data_config),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "lr0": lr0,
            "lrf": lrf,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "warmup_momentum": warmup_momentum,
            "warmup_bias_lr": warmup_bias_lr,
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "degrees": degrees,
            "translate": translate,
            "scale": scale,
            "shear": shear,
            "perspective": perspective,
            "flipud": flipud,
            "fliplr": fliplr,
            "mosaic": mosaic,
            "mixup": mixup,
            "copy_paste": copy_paste,
            "patience": patience,
            "save_period": save_period,
            "workers": workers,
            "project": str(self.project),
            "name": self.name,
            "exist_ok": self.exist_ok,
            "resume": resume,
            "amp": amp,
            "fraction": fraction,
        }

        if device is not None:
            train_args["device"] = device

        # Add any additional kwargs
        train_args.update(kwargs)

        # Train the model
        try:
            results = self.model.train(**train_args)
            logger.info("Training completed successfully!")

            # Get best model path
            best_model = self.project / self.name / "weights" / "best.pt"
            last_model = self.project / self.name / "weights" / "last.pt"

            logger.info(f"Best model saved to: {best_model}")
            logger.info(f"Last model saved to: {last_model}")

            return {
                "best_model": str(best_model),
                "last_model": str(last_model),
                "results": results,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def validate(self, data: Path | None = None, **kwargs) -> dict:
        """
        Validate the model.

        Args:
            data: Path to validation data config
            **kwargs: Additional validation arguments

        Returns:
            Dictionary containing validation results
        """
        data = data or self.data_config
        if data is None:
            raise ValueError("data must be provided for validation")

        logger.info(f"Validating model on {data}")
        results = self.model.val(data=str(data), **kwargs)

        return results


class Detectron2Trainer:
    """Trainer for Detectron2 models."""

    def __init__(
        self,
        config_file: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        weights: str | None = None,
        output_dir: Path = Path("output"),
    ):
        """
        Initialize Detectron2 trainer.

        Args:
            config_file: Path to config file or config name
            weights: Path to pretrained weights (or None for COCO weights)
            output_dir: Output directory for checkpoints
        """
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
        except ImportError:
            raise ImportError(
                "detectron2 package not installed. "
                "See: https://detectron2.readthedocs.io/en/latest/tutorials/install.html"
            ) from None

        self.cfg = get_cfg()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        if Path(config_file).exists():
            self.cfg.merge_from_file(config_file)
        else:
            self.cfg.merge_from_file(model_zoo.get_config_file(config_file))

        # Set weights
        if weights:
            self.cfg.MODEL.WEIGHTS = weights
        else:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

        self.cfg.OUTPUT_DIR = str(self.output_dir)

        logger.info(f"Initialized Detectron2 with config: {config_file}")

    def train(
        self,
        dataset_name: str,
        num_classes: int,
        max_iter: int = 5000,
        batch_size: int = 2,
        base_lr: float = 0.00025,
        checkpoint_period: int = 500,
        eval_period: int = 500,
        **kwargs,
    ) -> None:
        """
        Train Detectron2 model.

        Args:
            dataset_name: Name of registered dataset
            num_classes: Number of object classes
            max_iter: Maximum training iterations
            batch_size: Images per batch
            base_lr: Base learning rate
            checkpoint_period: Checkpoint save period
            eval_period: Evaluation period
            **kwargs: Additional config options
        """
        from detectron2.engine import DefaultTrainer

        self.cfg.DATASETS.TRAIN = (dataset_name,)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = batch_size
        self.cfg.SOLVER.BASE_LR = base_lr
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
        self.cfg.TEST.EVAL_PERIOD = eval_period
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        # Apply additional kwargs
        for key, value in kwargs.items():
            # Convert dot notation to nested config
            keys = key.split(".")
            cfg_node = self.cfg
            for k in keys[:-1]:
                cfg_node = getattr(cfg_node, k)
            setattr(cfg_node, keys[-1], value)

        logger.info("Starting Detectron2 training")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Max iterations: {max_iter}")
        logger.info(f"  Batch size: {batch_size}")

        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        logger.info("Training completed!")


def load_config(config_path: Path) -> dict:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, output_path: Path) -> None:
    """
    Save training configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train object detection models for GUI element detection"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8",
        choices=["yolov8", "detectron2"],
        help="Model architecture (default: yolov8)",
    )

    # YOLOv8 specific
    parser.add_argument(
        "--model-size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size (default: n)",
    )

    # Data configuration
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to data.yaml configuration file",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--lr0", type=float, default=0.01, help="Initial learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor (default: 0.01)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.937, help="SGD momentum (default: 0.937)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay (default: 0.0005)",
    )

    # Augmentation
    parser.add_argument(
        "--no-augment", action="store_true", help="Disable data augmentation"
    )
    parser.add_argument(
        "--mosaic", type=float, default=1.0, help="Mosaic augmentation (default: 1.0)"
    )
    parser.add_argument(
        "--mixup", type=float, default=0.0, help="MixUp augmentation (default: 0.0)"
    )
    parser.add_argument(
        "--degrees", type=float, default=0.0, help="Rotation degrees (default: 0.0)"
    )
    parser.add_argument(
        "--translate",
        type=float,
        default=0.1,
        help="Translation fraction (default: 0.1)",
    )
    parser.add_argument(
        "--scale", type=float, default=0.5, help="Scale gain (default: 0.5)"
    )
    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="Horizontal flip probability (default: 0.5)",
    )

    # Training control
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (default: 50)",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every n epochs (default: -1, disabled)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers (default: 8)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu/device id)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable automatic mixed precision"
    )

    # Output control
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/train"),
        help="Project directory (default: runs/train)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    parser.add_argument(
        "--exist-ok", action="store_true", help="Allow overwriting existing results"
    )

    # Config file support
    parser.add_argument(
        "--config", type=Path, help="Path to YAML config file (overrides CLI args)"
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        help="Save training config to YAML file and exit",
    )

    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)
        # Update args with config values (CLI args take precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    # Generate experiment name if not provided
    if args.name is None:
        timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
        args.name = f"{args.model}_{args.model_size}_{timestamp}"

    # Save config if requested
    if args.save_config:
        config = vars(args)
        save_config(config, args.save_config)
        logger.info(f"Config saved to {args.save_config}")
        return

    # Check CUDA availability
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "0"
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            args.device = "cpu"
            logger.warning("CUDA not available. Using CPU (training will be slow)")

    # Train based on model type
    if args.model == "yolov8":
        trainer = YOLOv8Trainer(
            model_size=args.model_size,
            data_config=args.data,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
        )

        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            augment=not args.no_augment,
            mosaic=args.mosaic,
            mixup=args.mixup,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            fliplr=args.fliplr,
            patience=args.patience,
            save_period=args.save_period,
            workers=args.workers,
            device=args.device,
            resume=args.resume,
            amp=not args.no_amp,
        )

        logger.info(f"Training results: {results}")

    elif args.model == "detectron2":
        logger.error("Detectron2 training not fully implemented yet")
        logger.info(
            "Please refer to Detectron2 documentation: "
            "https://detectron2.readthedocs.io/"
        )
        sys.exit(1)

    else:
        parser.error(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
