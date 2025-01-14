import os
import yaml
from ultralytics import YOLO

def create_yaml_config(dataset_path):
    """Create YAML configuration for CytologIA dataset"""
    classes_path = os.path.join(dataset_path, 'yolo/dataset/classes.txt')
    with open(classes_path, 'r') as f:
        classes = [line.strip().split(':')[1].strip() for line in f.readlines()]

    config = {
        'path': dataset_path,
        'train': os.path.join(dataset_path, 'yolo/dataset/images/train'),
        'val': os.path.join(dataset_path, 'yolo/dataset/images/val'),
        'test': os.path.join(dataset_path, 'yolo/dataset/images/test'),
        'nc': len(classes),
        'names': {i: name for i, name in enumerate(classes)},
        'overlap_threshold': 0.3
    }

    config_path = os.path.join('config', 'dataset_config.yaml')
    os.makedirs('config', exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path

def train_model(dataset_path, resume_path=None):
    """Train YOLOv8 model with optimized parameters"""
    yaml_path = create_yaml_config(dataset_path)
    model = YOLO('yolov8s.pt')

    results = model.train(
        data=yaml_path,
        epochs=80,
        patience=10,
        batch=8,
        imgsz=800,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        mixup=0.15,
        mosaic=0.8,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        degrees=45.0,
        perspective=0.0001,
        copy_paste=0.1,
        device=0,
        workers=8,
        project='runs/detect',
        name='train',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        dropout=0.2,
        label_smoothing=0.1,
        cos_lr=True,
        plots=True,
        save=True,
        save_period=5,
        resume=True if resume_path else False
    )
    return results

if __name__ == "__main__":
    dataset_path = ""  
    results = train_model(dataset_path)