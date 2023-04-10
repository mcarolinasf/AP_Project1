#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
teste
"""

from tp1_utils import load_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Load data for main types only
data = load_data()
x_train = data["train_X"]
y_train = data["train_classes"]
x_val = data["test_X"]
y_val = data["test_classes"]


# ** ----------------------------- MLP ----------------------------- **

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



# ** ----------------------------- CNN ----------------------------- **

# Hyperparameters
cnn_loss = 'categorical_crossentropy'
cnn_learning_rate = 0.0001
cnn_optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_learning_rate)
cnn_metrics = ['accuracy']
cnn_epochs = 50

# Create model
cnn_model = Sequential(
    [
      Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(64, 64, 3)),
      MaxPooling2D(pool_size=(2, 2)),
      Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
      MaxPooling2D(pool_size=(2, 2)),
      Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
      Flatten(),
      Dense(units=64, activation='relu'),
      Dense(units=10, activation='softmax')
    ]
)

# Compile
cnn_model.compile(optimizer=cnn_optimizer, loss=cnn_loss, metrics=cnn_metrics)

# Fit
history = cnn_model.fit(x=x_train, y=y_train, epochs=cnn_epochs, validation_data=(x_val, y_val))

# Evaluate
test_loss, test_acc = cnn_model.evaluate(x=x_val, y=y_val)
print("Validation loss:", test_loss)
print("Validation accuracy:", test_acc)



# ** ----------------------------- LOLOLOLOLOL ----------------------------- **


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
