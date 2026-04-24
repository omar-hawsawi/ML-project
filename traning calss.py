import pandas as pd
import numpy as np
import tensorflow as tf
import os

class SignLanguageTrainer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # Map the signs to numbers (250 unique signs)
        self.label_map = {sign: i for i, sign in enumerate(self.df['sign'].unique())}
        self.num_classes = len(self.label_map)

    def load_landmarks(self, parquet_path, max_len=30):
        try:
            data = pd.read_parquet(parquet_path)
            # Select x, y, z and fill missing values with 0
            coords = data[['x', 'y', 'z']].fillna(0).values
            
            # --- PADDING LOGIC ---
            # If too long, cut it. If too short, add zeros.
            if len(coords) > max_len:
                coords = coords[:max_len]
            else:
                pad_width = max_len - len(coords)
                coords = np.pad(coords, ((0, pad_width), (0, 0)), mode='constant')
            return coords
        except Exception as e:
            return np.zeros((max_len, 3))

    def data_generator(self, base_path='.', batch_size=8):
        while True:
            # Shuffle the dataframe every epoch
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            for idx in range(0, len(self.df), batch_size):
                X_batch, y_batch = [], []
                batch_df = self.df.iloc[idx : idx + batch_size]
                
                for _, row in batch_df.iterrows():
                    parquet_file = os.path.join(base_path, row['path'])
                    landmarks = self.load_landmarks(parquet_file)

                    # --- ADD THIS CHECK HERE ---
                    # It ensures every item in the batch is identical.
                    # Change (30, 3) to match whatever your max_len is!
                    if landmarks is not None and hasattr(landmarks, 'shape') and landmarks.shape == (30, 3):
                        X_batch.append(landmarks)
                        y_batch.append(self.label_map[row['sign']])

                if len(X_batch) > 0:
                    yield np.stack(X_batch), np.array(y_batch)

    def build_model(self, input_shape=(30, 3)):
        model = tf.keras.Sequential([
            # Input is (Time Steps, 3 Coordinates)
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model
