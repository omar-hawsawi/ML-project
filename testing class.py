import numpy as np
import pandas as pd
import tensorflow as tf
import os 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class SignLanguageTester:
    def __init__(self, csv_path, model_path):
        self.df = pd.read_csv(csv_path)
        #Create the same label map as in the trainer to ensure consistency
        self.unique_signs = sorted(self.df['sign'].unique())
        self.label_map = {sign: i for i, sign in enumerate(self.unique_signs)}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

         # Loading the trained TCN model
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

        def load_landmarks(self, parquet_path, max_len=30):
            """Must have the same logic as in the trainer class to ensure consistency."""
            try:
                data = pd.read_parquet(parquet_path)
                coords = data[['x', 'y', 'z']].fillna(0).values

                if len(coords) > max_len:
                    coords = coords[:max_len]
                else:
                    pad_width = max_len - len(coords)
                    coords = np.pad(coords, ((0, pad_width), (0, 0)), mode='constant')
                return coords
            except Exception as e:
                return np.zeros((max_len, 3))
            
    def evaluate(self, base_path='.'):
        X_test, y_true = [], []

        print("Preparing test data...")
        for _, row in self.df.iterrows():
            parquet_file = os.path.join(base_path, row['path'])
            landmarks = self.load_landmarks(parquet_file)
        
        # Ensure the landmarks are in the correct shape before adding to the test set
        if landmarks.shape == (30, 3):
           X_test.append(landmarks)
           y_true.append(self.label_map[row['sign']])  
        
        if not X_test:
            print("No valid test data found. Please check your dataset and paths.")
            return
        
        X_test = np.array(X_test)
        y_true = np.array(y_true)

        # Predicting with the loaded model
        print("Predicting...")
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Calculating metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n--- Test Results ---")
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Detailed classification report
        # zero_division=0 prevents the report from crashing the model if there are classes with no predicted samples.
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.unique_signs, zero_division=0))

        # Confusion Matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=list(self.label_map.values()))) 



#run the evaluation
if __name__ == "__main__":
 # Update these paths to your actual files!
 tester = SignLanguageTester(csv_path='train.csv', model_path='tcn_model.h5')
 tester.evaluate(base_path='.')

