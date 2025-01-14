import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetPreparation:
    @staticmethod
    def split_dataset(image_dir, labels_dir, train_size=0.8, val_size=0.1, random_state=42):
        """Split dataset into train, validation, and test sets"""
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        
        # First split: train + validation vs test
        train_val_files, test_files = train_test_split(
            image_files,
            train_size=train_size + val_size,
            random_state=random_state
        )
        
        # Second split: train vs validation
        relative_val_size = val_size / (train_size + val_size)
        train_files, val_files = train_test_split(
            train_val_files,
            train_size=(1 - relative_val_size),
            random_state=random_state
        )
        
        return train_files, val_files, test_files

    @staticmethod
    def organize_dataset(source_dir, dest_dir, split_files, split_name):
        """Organize dataset files into train/val/test directories"""
        images_dir = os.path.join(dest_dir, 'images', split_name)
        labels_dir = os.path.join(dest_dir, 'labels', split_name)
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        for file in split_files:
            # Copy images
            src_img = os.path.join(source_dir, 'images', file)
            dst_img = os.path.join(images_dir, file)
            if os.path.exists(src_img):
                os.system(f'cp "{src_img}" "{dst_img}"')
            
            # Copy labels (assuming .txt extension for label files)
            label_file = os.path.splitext(file)[0] + '.txt'
            src_label = os.path.join(source_dir, 'labels', label_file)
            dst_label = os.path.join(labels_dir, label_file)
            if os.path.exists(src_label):
                os.system(f'cp "{src_label}" "{dst_label}"')

class Visualizations:
    @staticmethod
    def plot_training_metrics(results_dir):
        """Plot training metrics from results directory"""
        metrics_file = os.path.join(results_dir, 'results.csv')
        if not os.path.exists(metrics_file):
            print(f"Results file not found: {metrics_file}")
            return
        
        metrics = pd.read_csv(metrics_file)
        
        plt.figure(figsize=(15, 10))
        
        # Plot training losses
        plt.subplot(2, 2, 1)
        plt.plot(metrics['epoch'], metrics['train/box_loss'], label='Box Loss')
        plt.plot(metrics['epoch'], metrics['train/cls_loss'], label='Class Loss')
        plt.plot(metrics['epoch'], metrics['train/dfl_loss'], label='DFL Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot metrics
        plt.subplot(2, 2, 2)
        plt.plot(metrics['epoch'], metrics['metrics/precision'], label='Precision')
        plt.plot(metrics['epoch'], metrics['metrics/recall'], label='Recall')
        plt.plot(metrics['epoch'], metrics['metrics/mAP50'], label='mAP50')
        plt.title('Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        
        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(metrics['epoch'], metrics['lr/pg0'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_metrics.png'))
        plt.close()

    @staticmethod
    def visualize_augmentations(image_path, num_examples=5):
        """Visualize data augmentations on a sample image"""
        from albumentations import (
            Compose, HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
            HueSaturationValue, RandomResizedCrop, GaussNoise
        )
        
        # Define augmentation pipeline
        transform = Compose([
            RandomResizedCrop(height=800, width=800, scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=45, p=0.5),
            RandomBrightnessContrast(p=0.5),
            HueSaturationValue(p=0.5),
            GaussNoise(p=0.5)
        ])
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create subplots
        plt.figure(figsize=(20, 4))
        
        # Original image
        plt.subplot(1, num_examples + 1, 1)
        plt.imshow(image)
        plt.title('Original')
        plt.axis('off')
        
        # Augmented images
        for i in range(num_examples):
            augmented = transform(image=image)['image']
            plt.subplot(1, num_examples + 1, i + 2)
            plt.imshow(augmented)
            plt.title(f'Augmented {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

class EvaluationMetrics:
    @staticmethod
    def calculate_metrics(predictions_df, ground_truth_df):
        """Calculate evaluation metrics for predictions"""
        def calculate_iou(box1, box2):
            # Calculate intersection over union of two boxes
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = box1_area + box2_area - intersection
            
            return intersection / union if union > 0 else 0

        metrics = {
            'correct_classifications': 0,
            'total_predictions': len(predictions_df),
            'average_iou': 0,
            'class_accuracies': {}
        }
        
        for _, pred_row in predictions_df.iterrows():
            gt_row = ground_truth_df[ground_truth_df['NAME'] == pred_row['NAME']].iloc[0]
            
            # Calculate IoU
            pred_box = [pred_row['x1'], pred_row['y1'], pred_row['x2'], pred_row['y2']]
            gt_box = [gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2']]
            iou = calculate_iou(pred_box, gt_box)
            metrics['average_iou'] += iou
            
            # Check classification
            if pred_row['Class'] == gt_row['Class']:
                metrics['correct_classifications'] += 1
                
                # Update class-specific accuracy
                if pred_row['Class'] not in metrics['class_accuracies']:
                    metrics['class_accuracies'][pred_row['Class']] = {'correct': 0, 'total': 0}
                metrics['class_accuracies'][pred_row['Class']]['correct'] += 1
            
            # Update class totals
            if pred_row['Class'] not in metrics['class_accuracies']:
                metrics['class_accuracies'][pred_row['Class']] = {'correct': 0, 'total': 0}
            metrics['class_accuracies'][pred_row['Class']]['total'] += 1
        
        # Calculate final metrics
        metrics['accuracy'] = metrics['correct_classifications'] / metrics['total_predictions']
        metrics['average_iou'] = metrics['average_iou'] / metrics['total_predictions']
        
        # Calculate per-class accuracies
        for class_name in metrics['class_accuracies']:
            class_stats = metrics['class_accuracies'][class_name]
            class_stats['accuracy'] = class_stats['correct'] / class_stats['total'] if class_stats['total'] > 0 else 0
        
        return metrics

    @staticmethod
    def plot_confusion_matrix(predictions_df, ground_truth_df):
        """Plot confusion matrix for predictions"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        y_true = ground_truth_df['Class']
        y_pred = predictions_df['Class']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        classes = sorted(list(set(y_true)))
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()