import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import argparse
from src.train import train_model
from src.inference import CytologiaInference

def main():
    parser = argparse.ArgumentParser(description='Cytologia Cell Classification')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Mode: train or predict')
    parser.add_argument('--dataset_path', help='Path to dataset directory')
    parser.add_argument('--model_path', help='Path to model weights')
    parser.add_argument('--image_path', help='Path to image for prediction')
    parser.add_argument('--test_csv', help='Path to test.csv for submission generation')
    parser.add_argument('--test_dir', help='Path to test images directory')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.dataset_path:
            raise ValueError("dataset_path is required for training")
        results = train_model(args.dataset_path)
        print("Training completed. Results saved in runs/detect/train/")
    
    elif args.mode == 'predict':
        if not args.model_path:
            raise ValueError("model_path is required for prediction")
        
        inference = CytologiaInference(args.model_path)
        
        if args.image_path:
            # Single image prediction
            inference.visualize_prediction(args.image_path, save_path="prediction.png")
        
        if args.test_csv and args.test_dir:
            # Generate submission
            submission_df = inference.generate_submission(args.test_csv, args.test_dir)
            submission_df.to_csv('submission.csv', index=False)
            print("Submission file generated: submission.csv")

if __name__ == "__main__":
    main()