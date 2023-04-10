#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
teste
"""


from tp1_utils import load_data
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

# Load data for main types only

data = load_data()

x_train = data["train_X"]
y_train = data["train_classes"]

x_val = data["test_X"]
y_val = data["test_classes"]


# Shuffle data

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

dataset = dataset.shuffle(buffer_size=len(x_train))

dataset = dataset.batch(32)


# Create and compile model

mlp_model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(64, 64, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

mlp_model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.experimental.Adadelta(learning_rate=1.0),
    metrics=["accuracy"],
)


# Train the model
history = mlp_model.fit(x_train, y_train, batch_size=32, epochs=25, validation_data=(x_val, y_val))

# Evaluate the model using model.evaluate()
test_scores = mlp_model.evaluate(x_val, y_val, verbose=2)
print("Validation loss:", test_scores[0])
print("Validation accuracy:", test_scores[1])

# Plot the training and validation accuracy over time
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

# Plot the training and validation loss over time
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()
