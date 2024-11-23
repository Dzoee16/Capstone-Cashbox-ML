import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
from data_processing import preprocess_training_data

def train_and_save_model(data_path, model_path):
    """
    Train and save the TensorFlow model.
    """
    X, y = preprocess_training_data(data_path)

    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation='sigmoid')  # Urgency score
    ])
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        # loss='mean_absolute_error',
        # metrics=['mae', 'mse']
        )

    # Train model
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

    # Save model
    model.save(model_path)

if __name__ == "__main__":
    train_and_save_model("data/financial_data.csv", "models/urgency_model.h5")
