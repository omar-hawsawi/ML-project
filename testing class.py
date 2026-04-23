import pandas as pd
import numpy as np
import tensorflow as tf
import os

class SignLanguageTrainer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.label_map = {sign: i for i, sign in enumerate(self.df['sign'].unique())}
        self.model = None
    
    def data_generator(self, base_path='.', batch_size=32):
        """Generator to yield batches of data"""
        while True:
            for idx in range(0, len(self.df), batch_size):
                X_batch, y_batch = [], []
                for i in range(idx, min(idx + batch_size, len(self.df))):
                    try:
                        row = self.df.iloc[i]
                        parquet_file = os.path.join(base_path, row['path'])
                        landmarks = self.load_landmarks(parquet_file)
                        X_batch.append(landmarks)
                        y_batch.append(self.label_map[row['sign']])
                    except:
                        continue
                yield np.array(X_batch), np.array(y_batch)
    def preparing_brane(self, parquet_path):
        num_classes = len(self.df['sign'].unique()) [cite: 1,5]
        model = trainer.build_model(input_shape=(None, 3), num_classes=num_classes)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_landmarks(self, parquet_path):
        
        try:
            data = pd.read_parquet(parquet_path)
            
            return data[['x', 'y', 'z']].fillna(0).values
        except:
            return np.zeros((1, 3)) 
            
    def build_model(self, input_shape, num_classes):
        model = tf.keras.Sequential([
           
            tf.keras.layers.Input(shape=input_shape),
            
            
            tf.keras.layers.LSTM(128, return_sequences=False),
            
           
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2), 
            tf.keras.layers.Dense(num_classes, activation='softmax') 
        ])
        return model




from traning_calss import SignLanguageTrainer

trainer = SignLanguageTrainer('train.csv')


model = trainer.preparing_brain()


train_gen = trainer.data_generator(batch_size=32)


print("Training is starting... Look at the 'accuracy' number!")
model.fit(
    train_gen,
    steps_per_epoch=100, 
    epochs=10           
)


model.save('ML_project.h5')
print("Finished! Model saved as ML_project.h5")
