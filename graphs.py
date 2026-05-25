import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from traning_calss import SignLanguageTrainer


def plot_loss_and_speed(history, duration, model_name, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], marker='o', color='tab:red')
    axes[0].set_title(f'{model_name} training loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].bar([model_name], [duration], color='tab:blue')
    axes[1].set_title(f'{model_name} training speed')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].text(0, duration + duration * 0.02, f'{duration:.1f}s', ha='center', va='bottom')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'Saved graph: {out_path}')


def build_callbacks():
    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=8, restore_best_weights=True, verbose=1
        )
    ]


def train_tcn(csv_path='train.csv', base_path='.', batch_size=16, epochs=30, steps_per_epoch=100):
    trainer = SignLanguageTrainer(csv_path)
    model = trainer.build_model()

    print('Training TCN model...')
    start_time = time.time()
    history = model.fit(
        trainer.data_generator(base_path=base_path, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=build_callbacks(),
        verbose=1
    )
    duration = time.time() - start_time

    final_loss = history.history['loss'][-1]
    print(f'Final TCN loss: {final_loss:.4f}')
    print(f'TCN training time: {duration:.1f} seconds')

    plot_loss_and_speed(history, duration, 'TCN', 'tcn_loss_and_speed.png')
    return model, history, duration


if __name__ == '__main__':
    csv_path = 'train.csv'
    base_path = '.'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    train_tcn(csv_path=csv_path, base_path=base_path, batch_size=16, epochs=30, steps_per_epoch=100)

