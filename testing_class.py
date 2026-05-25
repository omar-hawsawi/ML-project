import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

class SignLanguageTester:
    def __init__(self, model_path, csv_path, sign_map_path='sign_to_prediction_index_map.json'):
        """
        Initialize the tester with a trained TCN model
        
        Args:
            model_path: Path to the trained model (e.g., 'tcn_best.h5')
            csv_path: Path to the CSV file with test data
            sign_map_path: Path to the sign to index mapping
        """
        self.model = tf.keras.models.load_model(model_path)
        self.df = pd.read_csv(csv_path)
        self.label_map = {sign: i for i, sign in enumerate(self.df['sign'].unique())}
        self.reverse_label_map = {i: sign for sign, i in self.label_map.items()}
        self.num_classes = len(self.label_map)
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Testing data loaded from {csv_path}")
        print(f"✓ Number of sign classes: {self.num_classes}")

    def load_landmarks(self, parquet_path, max_len=30):
        """Load and pad/truncate landmarks from parquet file"""
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
            print(f"Error loading {parquet_path}: {e}")
            return np.zeros((max_len, 3))

    def test_single_sample(self, parquet_path):
        """
        Test a single sample and return prediction
        
        Args:
            parquet_path: Path to the parquet file
            
        Returns:
            dict with prediction, confidence, and top-3 predictions
        """
        landmarks = self.load_landmarks(parquet_path)
        landmarks = np.expand_dims(landmarks, axis=0)  # Add batch dimension
        
        # Get prediction
        predictions = self.model.predict(landmarks, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get top-3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = [(self.reverse_label_map[idx], float(predictions[0][idx])) 
                 for idx in top_3_indices]
        
        return {
            'predicted_sign': self.reverse_label_map[predicted_class],
            'confidence': float(confidence),
            'top_3_predictions': top_3
        }

    def test_batch(self, base_path='.', num_samples=None):
        """
        Test on a batch of samples from the CSV
        
        Args:
            base_path: Base path to the landmark files
            num_samples: Number of samples to test (None = all)
            
        Returns:
            dict with overall metrics
        """
        test_df = self.df.copy()
        if num_samples:
            test_df = test_df.head(num_samples)
        
        y_true = []
        y_pred = []
        
        print(f"\nTesting {len(test_df)} samples...")
        for idx, (_, row) in enumerate(test_df.iterrows()):
            if (idx + 1) % 50 == 0:
                print(f"  Processed: {idx + 1}/{len(test_df)}")
            
            parquet_file = os.path.join(base_path, row['path'])
            landmarks = self.load_landmarks(parquet_file)
            
            if landmarks is not None and landmarks.shape == (30, 3):
                landmarks = np.expand_dims(landmarks, axis=0)
                prediction = self.model.predict(landmarks, verbose=0)
                predicted_class = np.argmax(prediction[0])
                
                y_true.append(self.label_map[row['sign']])
                y_pred.append(predicted_class)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print("\n" + "="*60)
        print("TCN MODEL TEST RESULTS")
        print("="*60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

    def test_with_confidence_filter(self, base_path='.', confidence_threshold=0.7):
        """
        Test model but only count predictions with high confidence
        
        Args:
            base_path: Base path to landmark files
            confidence_threshold: Minimum confidence to count as correct
            
        Returns:
            dict with filtered test results
        """
        test_df = self.df.copy()
        
        y_true = []
        y_pred = []
        confidences = []
        high_confidence_count = 0
        
        print(f"\nTesting with confidence threshold: {confidence_threshold}")
        for idx, (_, row) in enumerate(test_df.iterrows()):
            if (idx + 1) % 50 == 0:
                print(f"  Processed: {idx + 1}/{len(test_df)}")
            
            parquet_file = os.path.join(base_path, row['path'])
            landmarks = self.load_landmarks(parquet_file)
            
            if landmarks is not None and landmarks.shape == (30, 3):
                landmarks = np.expand_dims(landmarks, axis=0)
                prediction = self.model.predict(landmarks, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                
                y_true.append(self.label_map[row['sign']])
                y_pred.append(predicted_class)
                confidences.append(confidence)
                
                if confidence >= confidence_threshold:
                    high_confidence_count += 1
        
        # Calculate accuracy only for high-confidence predictions
        high_conf_indices = np.where(np.array(confidences) >= confidence_threshold)[0]
        if len(high_conf_indices) > 0:
            y_true_high = np.array(y_true)[high_conf_indices]
            y_pred_high = np.array(y_pred)[high_conf_indices]
            accuracy_high = accuracy_score(y_true_high, y_pred_high)
        else:
            accuracy_high = 0
        
        print("\n" + "="*60)
        print("TCN MODEL TEST RESULTS (WITH CONFIDENCE FILTER)")
        print("="*60)
        print(f"Samples with confidence >= {confidence_threshold}: {high_confidence_count}/{len(y_true)}")
        print(f"Accuracy (high confidence only): {accuracy_high:.4f}")
        print(f"Average confidence: {np.mean(confidences):.4f}")
        print("="*60)
        
        return {
            'accuracy': accuracy_high,
            'high_confidence_count': high_confidence_count,
            'total_samples': len(y_true),
            'average_confidence': np.mean(confidences)
        }
