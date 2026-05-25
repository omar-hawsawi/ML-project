from traning_calss import SignLanguageTrainer
import tensorflow as tf
import os

# 1. Initialize the trainer with your CSV file
print("Loading training data...")
trainer = SignLanguageTrainer('train.csv')

# 2. Build and compile the TCN model
print(f"Building TCN model with {trainer.num_classes} classes...")
model = trainer.build_model()
model.summary()

# 3. Create the data generator for training
print("Preparing data generator...")
train_gen = trainer.data_generator(batch_size=8)

# 4. Define callbacks for better training
callbacks = [
    # Save the best model based on accuracy
    tf.keras.callbacks.ModelCheckpoint(
        'tcn_best.h5',
        monitor='accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Stop early if accuracy plateaus
    tf.keras.callbacks.EarlyStopping(
        monitor='accuracy',
        patience=3,
        verbose=1
    ),
    # Reduce learning rate if accuracy plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='accuracy',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# 5. START TRAINING THE TCN MODEL!
print("\n" + "="*60)
print("Training TCN Model - Starting...")
print("="*60)
history = model.fit(
    train_gen,
    steps_per_epoch=100,  # How many batches per epoch
    epochs=15,             # How many times to repeat the whole dataset
    callbacks=callbacks,
    verbose=1
)

# 6. Save the final trained model
print("\nSaving final model...")
model.save('ML_project.h5')
print("✓ Finished! Final model saved as ML_project.h5")
print("✓ Best model saved as tcn_best.h5")
print("\nTraining Summary:")
print(f"  - Final Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  - Total Epochs: {len(history.history['accuracy'])}")
