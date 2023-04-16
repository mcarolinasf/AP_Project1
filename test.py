
from tp1_utils import load_data
import tensorflow as tf
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import precision_score, recall_score
from utils import PlotRelevantInfo

# Load data for main types only
data = load_data()
x_train = data["train_X"]
y_train = data["train_classes"]
x_val = data["test_X"]
y_val = data["test_classes"]

def MLP():


def CNN():
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

def NN():

def DNN_Multiclass():

def DNN_Multilabel():


