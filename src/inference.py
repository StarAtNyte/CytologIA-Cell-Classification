import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt

class CytologiaInference:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_mapping = {
            0: "B", 1: "BA", 2: "EO", 3: "Er", 4: "LAM3", 5: "LF",
            6: "LGL", 7: "LH_lyAct", 8: "LLC", 9: "LM", 10: "LY",
            11: "LZMG", 12: "LyB", 13: "Lysee", 14: "M", 15: "MBL",
            16: "MM", 17: "MO", 18: "MoB", 19: "PM", 20: "PNN",
            21: "SS", 22: "Thromb"
        }

    def predict_single_image(self, image_path, conf_threshold=0.25):
        """Run inference on a single image"""
        results = self.model.predict(
            source=image_path,
            imgsz=800,
            conf=conf_threshold
        )
        return results[0]

    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction on an image"""
        results = self.predict_single_image(image_path)
        
        # Plot the results
        annotated_img = results.plot(line_width=1)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_img)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()

    def generate_submission(self, test_csv_path, test_image_dir):
        """Generate submission file for competition"""
        test_df = pd.read_csv(test_csv_path)
        submissions = []

        for _, row in test_df.iterrows():
            img_name = row['NAME']
            img_path = os.path.join(test_image_dir, img_name)

            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_name}")
                submissions.append({
                    'trustii_id': row['trustii_id'],
                    'NAME': img_name,
                    'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1,
                    'Class': 'B'
                })
                continue

            results = self.predict_single_image(img_path)

            if len(results.boxes) > 0:
                boxes = results.boxes
                best_idx = boxes.conf.argmax().item()
                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                class_id = int(boxes.cls[best_idx].item())
                predicted_class = self.class_mapping[class_id]

                submissions.append({
                    'trustii_id': row['trustii_id'],
                    'NAME': img_name,
                    'x1': round(x1), 'y1': round(y1),
                    'x2': round(x2), 'y2': round(y2),
                    'Class': predicted_class
                })
            else:
                submissions.append({
                    'trustii_id': row['trustii_id'],
                    'NAME': img_name,
                    'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1,
                    'Class': 'B'
                })

        submission_df = pd.DataFrame(submissions)
        return submission_df

if __name__ == "__main__":
    # Example usage
    model_path = "...best.pt"  # Update with your model path
    inference = CytologiaInference(model_path)
    
    # Single image prediction
    image_path = "yolo\dataset\images\test\0a0edafd-e.jpg"
    inference.visualize_prediction(image_path, save_path="prediction.png")
    
    # Generate submission
    test_csv = "../test.csv"
    test_dir = "../images"
    submission_df = inference.generate_submission(test_csv, test_dir)
    submission_df.to_csv('../submission.csv', index=False)