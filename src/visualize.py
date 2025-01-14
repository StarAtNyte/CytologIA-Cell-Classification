import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
import seaborn as sns
from PIL import Image

class ResultsVisualization:
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def plot_training_curves(self):
        """Plot training curves from training results"""
        results_file = os.path.join(self.results_dir, 'results.csv')
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            return

        results = pd.read_csv(results_file)
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Results', fontsize=16)
        
        # Plot loss curves
        axs[0, 0].plot(results['epoch'], results['train/box_loss'], label='Train Box Loss')
        axs[0, 0].plot(results['epoch'], results['val/box_loss'], label='Val Box Loss')
        axs[0, 0].set_title('Box Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        
        # Plot mAP curves
        axs[0, 1].plot(results['epoch'], results['metrics/mAP50'], label='mAP50')
        axs[0, 1].plot(results['epoch'], results['metrics/mAP50-95'], label='mAP50-95')
        axs[0, 1].set_title('Mean Average Precision')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('mAP')
        axs[0, 1].legend()
        
        # Plot precision-recall curves
        axs[1, 0].plot(results['epoch'], results['metrics/precision'], label='Precision')
        axs[1, 0].plot(results['epoch'], results['metrics/recall'], label='Recall')
        axs[1, 0].set_title('Precision and Recall')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Value')
        axs[1, 0].legend()
        
        # Plot learning rate
        axs[1, 1].plot(results['epoch'], results['lr/pg0'], label='Learning Rate')
        axs[1, 1].set_title('Learning Rate')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Learning Rate')
        axs[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'))
        plt.close()

    def plot_prediction_grid(self, model_path, test_dir, num_images=16):
        """Plot a grid of predictions on test images"""
        model = YOLO(model_path)
        
        # Get random test images
        test_images = os.listdir(test_dir)
        if len(test_images) > num_images:
            test_images = np.random.choice(test_images, num_images, replace=False)
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(len(test_images))))
        
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        fig.suptitle('Test Set Predictions', fontsize=16)
        
        for idx, img_name in enumerate(test_images):
            img_path = os.path.join(test_dir, img_name)
            
            # Run prediction
            results = model.predict(img_path)
            
            # Get annotated image
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Plot in grid
            row = idx // grid_size
            col = idx % grid_size
            axs[row, col].imshow(annotated_img)
            axs[row, col].axis('off')
            axs[row, col].set_title(f'Image {idx+1}')
        
        # Turn off any empty subplots
        for idx in range(len(test_images), grid_size * grid_size):
            row = idx // grid_size
            col = idx % grid_size
            axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'prediction_grid.png'))
        plt.close()

    def plot_class_distribution(self, predictions_df):
        """Plot distribution of predicted classes"""
        plt.figure(figsize=(15, 6))
        
        # Class distribution
        sns.countplot(data=predictions_df, x='Class')
        plt.xticks(rotation=45)
        plt.title('Distribution of Predicted Classes')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        plt.tight