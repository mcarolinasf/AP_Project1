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

class PlotRelevantInfo():
    def __init__(self, results):
        self.iterations = 10
        self.results = results
        self.titles = ['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss']

    def plot_results_and_print_means(self):
        for result, title in zip(self.results, self.titles):
            plt.plot(list(zip(*result)))
            plt.title(f'{title} over {self.iterations} iterations')
            plt.ylabel(title)
            plt.xlabel('Epoch')
            plt.show()
        for result, title in zip(self.results, self.titles):
            lasts = [r[-1] for r in result]
        print(f"Average {title}: {sum(lasts)/len(lasts)}")


tr_acc, val_acc, tr_loss, val_loss = [], [], [], []
for n in range(10):
    cnn_loss = 'categorical_crossentropy'
    cnn_metrics = ['accuracy']
    cnn_learning_rate = 0.001
    cnn_epochs = 50
    cnn_optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_learning_rate)
    cnn_model = Sequential(
        [
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3), kernel_regularizer=regularizers.l2(0.0001)),
            MaxPooling2D(pool_size=(3, 3)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
            MaxPooling2D(pool_size=(3, 3)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            MaxPooling2D(pool_size=(3, 3)),
            Flatten(),    
            Dense(units=10, activation='softmax')
        ]
    )
    cnn_model.compile(optimizer=cnn_optimizer, loss=cnn_loss, metrics=cnn_metrics)
    history = cnn_model.fit(x=x_train, y=y_train, epochs=cnn_epochs, validation_data=(x_val, y_val), batch_size=32, verbose=0)
    tr_acc.append(history.history['accuracy'])
    val_acc.append(history.history['val_accuracy'])
    tr_loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])
    print(f"{n+1}: {tr_acc[n][-1]}, {val_acc[n][-1]}, {tr_loss[n][-1]}, {val_loss[n][-1]}")

plot_relevant_data = PlotRelevantInfo((tr_acc, val_acc, tr_loss, val_loss))
plot_relevant_data.plot_results_and_print_means()


# ** ----------------------------- NN ----------------------------- **

# Hyperparameters
nn_loss = 'binary_crossentropy'
nn_learning_rate = 0.003
nn_optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_learning_rate)
nn_metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
nn_epochs = 50
nn_batch_size = 32

# Create model
nn_model = Sequential(
    [
      Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
      MaxPooling2D(pool_size=(2, 2)),
      Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
      MaxPooling2D(pool_size=(2, 2)),
      Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
      MaxPooling2D(pool_size=(2, 2)),
      Flatten(),
      Dense(units=128, activation='relu'),
      Dense(units=10, activation='sigmoid')
    ]
)

nn_model.compile(optimizer=nn_optimizer, loss=nn_loss, metrics=nn_metrics)

history = nn_model.fit(x_train, y_train, batch_size = nn_batch_size, epochs=nn_epochs, validation_data=(x_val, y_val))

test_scores = nn_model.evaluate(x_val, y_val, verbose=2)
print("Validation loss:", test_scores[0])
print("Validation accuracy:", test_scores[1])
print("Validation recall:", test_scores[2])
print("Validation precision:", test_scores[3])



# ** ----------------------------- Visual ----------------------------- **

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
