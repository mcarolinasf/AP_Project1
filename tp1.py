#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
teste
"""

from tp1_utils import load_data
import tensorflow as tf
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from keras.applications import VGG16, ResNet50

from utils import PlotRelevantInfo, PlotData


# Load data
data = load_data()

# ** ----------------------------- Models ----------------------------- **
def MLP():

    x_train = data["train_X"]
    y_train = data["train_classes"]
    x_val = data["test_X"]
    y_val = data["test_classes"]

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(x_train))
    dataset = dataset.batch(256)

    mlp_loss = "categorical_crossentropy"
    mlp_learning_rate = 0.0001
    mlp_optimizer = tf.keras.optimizers.Adam(learning_rate=mlp_learning_rate)
    mlp_metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    mlp_epochs = 75

    mlp_model = Sequential(
        [
            Flatten(),
            Dense(128, activation="relu", input_shape=(64, 64, 3), kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.3),
            Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.005)),
            Dropout(0.2),
            Dense(10, activation="softmax"),
        ]
    )

    mlp_model.compile(optimizer=mlp_optimizer, loss=mlp_loss, metrics=mlp_metrics)

    history = mlp_model.fit(dataset, epochs=mlp_epochs, validation_data=(x_val, y_val))
    test_scores = mlp_model.evaluate(x_val, y_val, verbose=2)
    print("Validation loss:", test_scores[0])
    print("Validation accuracy:", test_scores[1])
    print("Validation precision:", test_scores[2])
    print("Validation recall:", test_scores[3])
    
    PlotData(history)
                                                                          
def CNN():

    # Assign data
    x_train = data["train_X"]
    y_train = data["train_classes"]
    x_val = data["test_X"]
    y_val = data["test_classes"]

    cnn_val_acc, cnn_val_loss, cnn_precisions_per_epoch, cnn_recalls_per_epoch = [], [], [], []
    for n in range(10):
        cnn_loss = "categorical_crossentropy"
        cnn_learning_rate = 0.001
        cnn_optimizer = tf.keras.optimizers.Adam(learning_rate=cnn_learning_rate)
        cnn_metrics = ["accuracy"]
        cnn_epochs = 50

        cnn_model = Sequential(
            [
                Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    activation="relu",
                    input_shape=(64, 64, 3),
                    kernel_regularizer=regularizers.l2(0.0001),
                ),
                MaxPooling2D(pool_size=(3, 3)),
                Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.0005)),
                MaxPooling2D(pool_size=(3, 3)),
                Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                MaxPooling2D(pool_size=(3, 3)),
                Flatten(),
                Dense(units=10, activation="softmax"),
            ]
        )

        cnn_model.compile(optimizer=cnn_optimizer, loss=cnn_loss, metrics=cnn_metrics)
        history = cnn_model.fit(
            x=x_train, y=y_train, epochs=cnn_epochs, validation_data=(x_val, y_val), batch_size=32, verbose=0
        )

        cnn_val_acc.append(history.history["val_accuracy"])
        cnn_val_loss.append(history.history["val_loss"])

        print(f"{n+1}: {cnn_val_acc[n][-1]}, {cnn_val_loss[n][-1]}")

        y_pred = cnn_model.predict(x_val, verbose=0)
        y_pred_labels = np.zeros_like(y_pred)
        for a, b in zip(y_pred, y_pred_labels):
            b[np.argmax(a)] = 1

        cnn_precisions_per_epoch.append(precision_score(y_val, y_pred_labels, average=None))
        cnn_recalls_per_epoch.append(recall_score(y_val, y_pred_labels, average=None))

    plot_relevant_data = PlotRelevantInfo((cnn_val_acc, cnn_val_loss, cnn_precisions_per_epoch, cnn_recalls_per_epoch))
    plot_relevant_data.plot_results_and_print_means(cnn_model)

def NN():

    # Assign data
    x_train = data["train_X"]
    y_train = data["train_labels"]
    x_val = data["test_X"]
    y_val = data["test_labels"]

    nn_val_acc, nn_val_loss, nn_precisions_per_epoch, nn_recalls_per_epoch = [], [], [], []

    nn_batch_size = 64
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(x_train))
    dataset = dataset.batch(nn_batch_size)

    for n in range(1):
        nn_loss = "binary_crossentropy"
        nn_learning_rate = 0.001
        nn_optimizer = tf.keras.optimizers.Adam(learning_rate=nn_learning_rate)
        nn_metrics = ["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        nn_epochs = 50

        nn_model = Sequential(
            [
                Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(64, 64, 3)),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(units=128, activation="relu"),
                Dense(units=10, activation="sigmoid"),
            ]
        )

        nn_model.compile(optimizer=nn_optimizer, loss=nn_loss, metrics=nn_metrics)
        # history = nn_model.fit(x=x_train, y=y_train, epochs=nn_epochs, validation_data=(x_val, y_val), verbose=0)
        history = nn_model.fit(dataset, epochs=nn_epochs, validation_data=(x_val, y_val), verbose=0)

        nn_val_acc.append(history.history["val_accuracy"])
        nn_val_loss.append(history.history["val_loss"])

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

def DNN_Multiclass():

    dnn_mc_loss = "categorical_crossentropy"
    dnn_mc_learning_rate = 0.0001
    dnn_mc_optimizer = tf.keras.optimizers.Adam(learning_rate=dnn_mc_learning_rate)
    dnn_mc_metrics = ['accuracy']
    dnn_mc_epochs = 20

    # Assign data
    x_train = data["train_X"]
    y_train = data["train_classes"]
    x_val = data["test_X"]
    y_val = data["test_classes"]

    

    # Load the pre-trained VGG16 model
    vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

    # Model
    dnn_mc_model = Sequential()
    dnn_mc_model.add(vgg_model)
    dnn_mc_model.add(Flatten())
    dnn_mc_model.add(Dense(128, activation="relu"))
    dnn_mc_model.add(Dense(10, activation="softmax"))

    # Compile
    dnn_mc_model.compile(optimizer=dnn_mc_optimizer, loss=dnn_mc_loss, metrics=dnn_mc_metrics)

    # Train the model
    history = dnn_mc_model.fit(x_train, y_train, epochs=dnn_mc_epochs, validation_data=(x_val, y_val))

    test_scores = dnn_mc_model.evaluate(x_val, y_val, verbose=2)
    print("Validation loss:", test_scores[0])
    print("Validation accuracy:", test_scores[1])

    PlotData(history)

def DNN_Multilabel():

    dnn_ml_loss = "binary_crossentropy"
    dnn_ml_learning_rate = 0.0001
    dnn_ml_optimizer = tf.keras.optimizers.Adam(learning_rate=dnn_ml_learning_rate)
    dnn_ml_metrics = ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    dnn_ml_epochs = 30

    # Assign data
    x_train = data["train_X"]
    y_train = data["train_labels"]
    x_val = data["test_X"]
    y_val = data["test_labels"]

    # Load the pre-trained VGG16 model
    vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

    # Model
    dnn_ml_model = Sequential()
    dnn_ml_model.add(vgg_model)
    dnn_ml_model.add(Flatten())
    dnn_ml_model.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.0003)))
    dnn_ml_model.add(Dense(10, activation="sigmoid"))

    # Compile
    dnn_ml_model.compile(
        optimizer=dnn_ml_optimizer,
        loss=dnn_ml_loss,
        metrics=dnn_ml_metrics,
    )

    # Train the model
    history = dnn_ml_model.fit(x_train, y_train, epochs=dnn_ml_epochs, validation_data=(x_val, y_val))

    test_scores = dnn_ml_model.evaluate(x_val, y_val, verbose=2)
    print("Validation loss:", test_scores[0])
    print("Validation accuracy:", test_scores[1])
    print("Validation precision:", test_scores[2])
    print("Validation recall:", test_scores[3])

    PlotData(history)


# ** ----------------------------- Main ----------------------------- **

#Run all the models
MLP()
CNN()
NN()
DNN_Multiclass()
DNN_Multilabel()


