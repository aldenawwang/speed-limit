import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import shutil
from datetime import datetime

class SpeedSignTrainer:
    def __init__(self, 
                 data_yaml='data.yaml',
                 model_size='n',  # n, s, m, l, x
                 project_name='speed_sign_detector'):
        """
        Initialize the trainer
        
        Args:
            data_yaml: Path to data configuration file
            model_size: YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            project_name: Name for this training project
        """
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.project_name = project_name
        self.model = None
        
        # Check for GPU/MPS (Apple Silicon)
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"Using device: {self.device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            print(f"Using device: {self.device} (Apple Metal)")
            print(f"PyTorch version: {torch.__version__}")
            print("✓ MPS acceleration enabled for Apple Silicon")
        else:
            self.device = 'cpu'
            print(f"Using device: {self.device}")
            print("WARNING: No GPU acceleration available")
    
    def create_data_yaml(self, 
                         data_path='./data',
                         train_path='images/train',
                         val_path='images/val',
                         test_path=None,
                         class_names=None):
        """
        Create data.yaml configuration file
        
        Args:
            data_path: Root path to dataset
            train_path: Relative path to training images
            val_path: Relative path to validation images
            test_path: Optional path to test images
            class_names: List of class names
        """
        if class_names is None:
            class_names = ['speed_limit', 'construction', 'school_zone']
        
        data_config = {
            'path': data_path,
            'train': train_path,
            'val': val_path,
            'nc': len(class_names),
            'names': class_names
        }
        
        if test_path:
            data_config['test'] = test_path
        
        # Save yaml file
        with open(self.data_yaml, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"Created {self.data_yaml}")
        print(f"Classes: {class_names}")
        return data_config
    
    def verify_dataset(self):
        """Verify dataset structure and count images"""
        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        data_path = Path(config['path'])
        train_path = data_path / config['train']
        val_path = data_path / config['val']
        
        print("\n=== Dataset Verification ===")
        print(f"Data root: {data_path}")
        print(f"Classes ({config['nc']}): {config['names']}")
        
        # Count training images
        if train_path.exists():
            train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
            print(f"\nTraining images: {len(train_images)}")
            
            # Check for corresponding labels
            label_path = data_path / 'labels' / 'train'
            if label_path.exists():
                train_labels = list(label_path.glob('*.txt'))
                print(f"Training labels: {len(train_labels)}")
            else:
                print(f"WARNING: Label directory not found: {label_path}")
        else:
            print(f"ERROR: Training path not found: {train_path}")
            return False
        
        # Count validation images
        if val_path.exists():
            val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
            print(f"\nValidation images: {len(val_images)}")
            
            label_path = data_path / 'labels' / 'val'
            if label_path.exists():
                val_labels = list(label_path.glob('*.txt'))
                print(f"Validation labels: {len(val_labels)}")
            else:
                print(f"WARNING: Label directory not found: {label_path}")
        else:
            print(f"ERROR: Validation path not found: {val_path}")
            return False
        
        # Minimum dataset size check
        min_train = 100
        min_val = 20
        
        if len(train_images) < min_train:
            print(f"\nWARNING: Less than {min_train} training images. Consider collecting more data.")
        if len(val_images) < min_val:
            print(f"WARNING: Less than {min_val} validation images. Consider collecting more data.")
        
        print("\n✓ Dataset structure verified")
        return True
    
    def train(self,
              epochs=100,
              imgsz=640,
              batch=16,
              patience=20,
              save_period=10,
              pretrained=True,
              freeze_layers=0,
              augment=True,
              mosaic=1.0,
              mixup=0.0,
              copy_paste=0.0,
              degrees=10.0,
              translate=0.1,
              scale=0.5,
              shear=2.0,
              perspective=0.0,
              flipud=0.0,
              fliplr=0.5,
              hsv_h=0.015,
              hsv_s=0.7,
              hsv_v=0.4,
              resume=False,
              optimizer='auto',
              lr0=0.01,
              momentum=0.937,
              weight_decay=0.0005):
        """
        Train the YOLO model
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size (adjust based on GPU memory)
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            pretrained: Use pretrained weights
            freeze_layers: Number of layers to freeze
            augment: Enable augmentation
            mosaic: Mosaic augmentation probability
            mixup: Mixup augmentation probability
            copy_paste: Copy-paste augmentation probability
            degrees: Rotation augmentation degrees
            translate: Translation augmentation
            scale: Scale augmentation
            shear: Shear augmentation
            perspective: Perspective augmentation
            flipud: Vertical flip probability
            fliplr: Horizontal flip probability
            hsv_h: HSV-Hue augmentation
            hsv_s: HSV-Saturation augmentation
            hsv_v: HSV-Value augmentation
            resume: Resume from last checkpoint
            optimizer: Optimizer choice
            lr0: Initial learning rate
            momentum: SGD momentum
            weight_decay: Weight decay
        """
        # Load model
        if pretrained:
            model_name = f'yolov8{self.model_size}.pt'
            print(f"\nLoading pretrained model: {model_name}")
        else:
            model_name = f'yolov8{self.model_size}.yaml'
            print(f"\nTraining from scratch: {model_name}")
        
        self.model = YOLO(model_name)
        
        # Verify dataset before training
        if not self.verify_dataset():
            print("\nDataset verification failed. Please fix issues before training.")
            return None
        
        print(f"\n=== Starting Training ===")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}")
        print(f"Batch size: {batch}")
        print(f"Device: {self.device}")
        print(f"Augmentation: {augment}")
        
        # Training arguments
        train_args = {
            'data': self.data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': self.device,
            'patience': patience,
            'save_period': save_period,
            'project': 'runs/train',
            'name': self.project_name,
            'exist_ok': True,
            'pretrained': pretrained,
            'optimizer': optimizer,
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': resume,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'freeze': freeze_layers,
            'lr0': lr0,
            'lrf': 0.01,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
        }
        
        # Augmentation parameters
        if augment:
            train_args.update({
                'hsv_h': hsv_h,
                'hsv_s': hsv_s,
                'hsv_v': hsv_v,
                'degrees': degrees,
                'translate': translate,
                'scale': scale,
                'shear': shear,
                'perspective': perspective,
                'flipud': flipud,
                'fliplr': fliplr,
                'mosaic': mosaic,
                'mixup': mixup,
                'copy_paste': copy_paste,
            })
        
        # Start training
        print(f"\nTraining started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        results = self.model.train(**train_args)
        
        print(f"\n✓ Training completed!")
        print(f"Results saved to: runs/train/{self.project_name}")
        
        return results
    
    def validate(self, weights_path=None):
        """
        Validate the trained model
        
        Args:
            weights_path: Path to weights file (if None, uses last trained)
        """
        if weights_path:
            self.model = YOLO(weights_path)
        elif self.model is None:
            print("No model loaded. Please train or provide weights path.")
            return None
        
        print("\n=== Running Validation ===")
        metrics = self.model.val(
            data=self.data_yaml,
            imgsz=640,
            batch=16,
            device=self.device,
            plots=True
        )
        
        print("\n=== Validation Results ===")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def export_model(self, 
                     weights_path=None,
                     formats=['onnx', 'torchscript'],
                     imgsz=640,
                     optimize=True):
        """
        Export model to different formats
        
        Args:
            weights_path: Path to weights file
            formats: List of export formats
            imgsz: Image size for export
            optimize: Optimize exported model
        """
        if weights_path:
            self.model = YOLO(weights_path)
        elif self.model is None:
            print("No model loaded. Please train or provide weights path.")
            return
        
        print(f"\n=== Exporting Model ===")
        
        for fmt in formats:
            print(f"Exporting to {fmt}...")
            try:
                self.model.export(
                    format=fmt,
                    imgsz=imgsz,
                    optimize=optimize,
                    half=False if fmt == 'onnx' else True
                )
                print(f"✓ {fmt} export successful")
            except Exception as e:
                print(f"✗ {fmt} export failed: {e}")
    
    def test_inference(self, 
                       test_image,
                       weights_path=None,
                       conf=0.5,
                       save_path='test_result.jpg'):
        """
        Test model on a single image
        
        Args:
            test_image: Path to test image
            weights_path: Path to weights file
            conf: Confidence threshold
            save_path: Where to save result
        """
        if weights_path:
            self.model = YOLO(weights_path)
        elif self.model is None:
            print("No model loaded. Please train or provide weights path.")
            return
        
        print(f"\n=== Testing Inference ===")
        print(f"Image: {test_image}")
        print(f"Confidence threshold: {conf}")
        
        results = self.model.predict(
            source=test_image,
            conf=conf,
            save=True,
            device=self.device
        )
        
        print(f"✓ Results saved")
        return results
    
    def create_training_report(self, run_path):
        """
        Create a summary report of training results
        
        Args:
            run_path: Path to training run directory
        """
        run_path = Path(run_path)
        
        if not run_path.exists():
            print(f"Run path not found: {run_path}")
            return
        
        print(f"\n=== Training Report ===")
        print(f"Run directory: {run_path}")
        
        # Check for results file
        results_csv = run_path / 'results.csv'
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            print("\nFinal Metrics:")
            last_row = df.iloc[-1]
            metrics_to_show = ['metrics/precision(B)', 'metrics/recall(B)', 
                              'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
            
            for metric in metrics_to_show:
                if metric in df.columns:
                    print(f"{metric}: {last_row[metric]:.4f}")
        
        # List weights
        weights_dir = run_path / 'weights'
        if weights_dir.exists():
            print(f"\nSaved weights:")
            for weight_file in weights_dir.glob('*.pt'):
                size = weight_file.stat().st_size / (1024 * 1024)  # MB
                print(f"  - {weight_file.name} ({size:.1f} MB)")
        
        print(f"\nTo use the trained model:")
        print(f"  best_model = YOLO('{run_path}/weights/best.pt')")


def main():
    """Example usage with different training scenarios"""
    
    # Initialize trainer
    trainer = SpeedSignTrainer(
        data_yaml='data.yaml',
        model_size='n',  # Use 'n' for fast training, 's' or 'm' for better accuracy
        project_name='speed_sign_v1'
    )
    
    # Option 1: Create data.yaml if it doesn't exist
    if not os.path.exists('data.yaml'):
        print("Creating data.yaml configuration...")
        trainer.create_data_yaml(
            data_path='./data',
            train_path='images/train',
            val_path='images/val',
            class_names=['speed_limit', 'construction', 'school_zone']
        )
    
    # Option 2: Quick training (for testing)
    print("\n--- Quick Training Mode ---")
    results = trainer.train(
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        augment=True
    )
    
    # Option 3: Full training with heavy augmentation (production)
    # print("\n--- Production Training Mode ---")
    # results = trainer.train(
    #     epochs=200,
    #     imgsz=640,
    #     batch=16,
    #     patience=30,
    #     augment=True,
    #     mosaic=1.0,
    #     mixup=0.1,
    #     degrees=15.0,
    #     scale=0.7,
    #     hsv_h=0.02,
    #     hsv_s=0.8,
    #     hsv_v=0.5
    # )
    
    # Validate
    print("\n--- Validation ---")
    metrics = trainer.validate()
    
    # Export model
    print("\n--- Exporting Model ---")
    trainer.export_model(
        weights_path=f'runs/train/speed_sign_v1/weights/best.pt',
        formats=['onnx', 'torchscript']
    )
    
    # Test on sample image (if available)
    # trainer.test_inference(
    #     test_image='test_images/sample.jpg',
    #     weights_path='runs/train/speed_sign_v1/weights/best.pt',
    #     conf=0.5
    # )
    
    # Generate report
    trainer.create_training_report('runs/train/speed_sign_v1')
    
    print("\n=== Training Complete ===")
    print("Next steps:")
    print("1. Review training plots in runs/train/speed_sign_v1/")
    print("2. Test the model on new images")
    print("3. Integrate best.pt into your detector")


if __name__ == "__main__":
    main()