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
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


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
    self.acc_and_loss = results[:2]
    self.precision_and_recall = results[2:]
    self.titles = ['Validation Accuracy', 'Validation Loss', 'Precision', 'Recall']

  def plot_results_and_print_means(self, model):
    self.__plot_evolution_of_validation_acc_and_loss()
    self.__plot_avg_precision_and_recall_for_each_class()
    self.__print_avg_validation_accuracy_and_loss()

  def __plot_evolution_of_validation_acc_and_loss(self):
    for result, title in zip(self.acc_and_loss, self.titles[:2]):
      plt.plot(list(zip(*result)))
      plt.xlabel('Epoch')
      plt.ylabel(title)
      plt.title(f'{title} over {self.iterations} iterations')
      plt.show()

  def __plot_avg_precision_and_recall_for_each_class(self):
    for avg, title in zip(self.precision_and_recall, self.titles[2:]):
      avg = np.mean(avg, axis=0)
      plt.bar(np.arange(10), avg)
      plt.xticks(np.arange(10))
      plt.xlabel('Class')
      plt.ylabel(f'Avg {title}')
      plt.title(f'{title} over {self.iterations} iterations')
      plt.show() 
    
  def __print_avg_validation_accuracy_and_loss(self):
    for result, title in zip(self.acc_and_loss, self.titles[:2]):
      lasts = [r[-1] for r in result]
      print(f"Average {title}: {sum(lasts)/len(lasts)}")
      

cnn_val_acc, cnn_val_loss, cnn_precisions_per_epoch, cnn_recalls_per_epoch = [], [], [], []
for n in range(10):
  cnn_loss = 'categorical_crossentropy'
  cnn_learning_rate = 0.001
  cnn_optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_learning_rate)
  cnn_metrics = ['accuracy']
  cnn_epochs = 50

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

  cnn_val_acc.append(history.history['val_accuracy'])
  cnn_val_loss.append(history.history['val_loss'])

  print(f"{n+1}: {cnn_val_acc[n][-1]}, {cnn_val_loss[n][-1]}")

  y_pred = cnn_model.predict(x_val, verbose=0)
  y_pred_labels = np.zeros_like(y_pred)
  for a, b in zip(y_pred, y_pred_labels):
    b[np.argmax(a)] = 1

  cnn_precisions_per_epoch.append(precision_score(y_val, y_pred_labels, average=None))
  cnn_recalls_per_epoch.append(recall_score(y_val, y_pred_labels, average=None))

plot_relevant_data = PlotRelevantInfo((cnn_val_acc, cnn_val_loss, cnn_precisions_per_epoch, cnn_recalls_per_epoch))
plot_relevant_data.plot_results_and_print_means(cnn_model)


# ** ----------------------------- NN ----------------------------- **

nn_val_acc, nn_val_loss, nn_precisions_per_epoch, nn_recalls_per_epoch = [], [], [], []

nn_batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size=len(x_train))
dataset = dataset.batch(nn_batch_size)

for n in range(1):
  nn_loss = 'binary_crossentropy'
  nn_learning_rate = 0.001
  nn_optimizer = tf.keras.optimizers.Adam(learning_rate=nn_learning_rate)
  nn_metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
  nn_epochs = 50

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
  #history = nn_model.fit(x=x_train, y=y_train, epochs=nn_epochs, validation_data=(x_val, y_val), verbose=0)
  history = nn_model.fit(dataset, epochs=nn_epochs, validation_data=(x_val, y_val), verbose=0)

  nn_val_acc.append(history.history['val_accuracy'])
  nn_val_loss.append(history.history['val_loss'])

  print(f"{n+1}: {nn_val_acc[n][-1]}, {nn_val_loss[n][-1]}")

  y_pred = nn_model.predict(x_val, verbose=0)
  y_pred_labels = np.zeros_like(y_pred)
  for a, b in zip(y_pred, y_pred_labels):
    for i in np.argsort(a)[-2:]:
      b[i] = 1

  nn_precisions_per_epoch.append(precision_score(y_val, y_pred_labels, average=None))
  nn_recalls_per_epoch.append(recall_score(y_val, y_pred_labels, average=None))

plot_relevant_data = PlotRelevantInfo((nn_val_acc, nn_val_loss, nn_precisions_per_epoch, nn_recalls_per_epoch))
plot_relevant_data.plot_results_and_print_means(nn_model)


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
