
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_PATH = 'hand_dataset.npz'
# Allow overriding output name per run via TRAIN_RUN env var
TRAIN_RUN = os.environ.get('TRAIN_RUN', '')
MODEL_OUT = f'hand_tcn_best{("_run"+TRAIN_RUN) if TRAIN_RUN else ""}.h5'
# Allow overriding which model to load before training
LOAD_MODEL = os.environ.get('LOAD_MODEL', 'hand_tcn_best_run3.h5')
# Allow repeating training multiple times, saving sequential run files
# Default to 5 automatic training repeats and start run numbering at 1
TRAIN_REPEAT = int(os.environ.get('TRAIN_REPEAT', '5'))
START_RUN = int(os.environ.get('START_RUN', '1'))
# Allow overriding hyperparams via env
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '128'))
EPOCHS = int(os.environ.get('EPOCHS', '80'))
VALIDATION_SPLIT = float(os.environ.get('VALIDATION_SPLIT', '0.1'))


def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.LayerNormalization()(inputs)
    x = layers.Conv1D(64, kernel_size=1, padding='same', activation='relu')(x)
    dilations = [1,2,4,8]
    for d in dilations:
        residual = x
        conv = layers.Conv1D(64, kernel_size=3, padding='same', dilation_rate=d)(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Dropout(0.3)(conv)
        x = layers.Add()([residual, conv])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    data = np.load(DATA_PATH)
    X = data['X']
    y = data['y']
    print('Loaded', X.shape, y.shape)
    num_classes = int(np.max(y) + 1) if len(y)>0 else 0
    if num_classes == 0:
        raise SystemExit('No data to train on')
    # Simple per-sample normalization (per-feature) to center coords
    X_mean = X.reshape(X.shape[0], -1).mean(axis=1)
    # Normalize each sample independently by subtracting mean of x,y coords
    X = X.astype('float32')
    # Scale coordinates to zero-centered by subtracting 0.5 (image-normalized landmarks)
    X = X - 0.5

    current_model_path = LOAD_MODEL
    current_run = START_RUN
    last_saved_path = None

    for repeat_index in range(TRAIN_REPEAT):
        print(f"\n=== Training round {repeat_index + 1}/{TRAIN_REPEAT} ===")
        if os.path.exists(current_model_path):
            print(f"Loading previous model from {current_model_path}...")
            model = tf.keras.models.load_model(current_model_path)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print("Continuing training from previous checkpoint!")
        else:
            print(f"No previous model found at {current_model_path}. Starting from scratch.")
            model = build_model((X.shape[1], X.shape[2]), num_classes)

        model.summary()

        # Build callback list and target output path for this round
        callbacks = []
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

        if TRAIN_REPEAT > 1:
            model_out = f'hand_tcn_best_run{current_run}.h5'
        else:
            model_out = MODEL_OUT

        callbacks.append(ModelCheckpoint(model_out, monitor='val_loss', save_best_only=True, verbose=1))
        es_patience = int(os.environ.get('PATIENCE', '10'))
        rlp_patience = int(os.environ.get('RLP_PATIENCE', '4'))
        callbacks.append(EarlyStopping(monitor='val_loss', patience=es_patience, restore_best_weights=True, verbose=1))
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=rlp_patience, verbose=1))

        # Class weights to partially address imbalance
        unique, counts = np.unique(y, return_counts=True)
        class_weights = {int(u): float(np.median(counts) / c) for u, c in zip(unique, counts)}

        print(f"Training with EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, VALIDATION_SPLIT={VALIDATION_SPLIT}")
        start = time.time()
        history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT,
                            callbacks=callbacks, shuffle=True, class_weight=class_weights)
        duration = time.time() - start
        print('Training finished in %.1fs' % duration)
        print('Saved best model to', model_out)
        last_saved_path = model_out
        current_model_path = model_out
        current_run += 1

    if last_saved_path is not None:
        print('\nFinal model saved to', last_saved_path)
    if 'val_loss' in history.history:
        print('final val_loss:', history.history['val_loss'][-1])
    if 'val_accuracy' in history.history:
        print('final val_accuracy:', history.history['val_accuracy'][-1])
