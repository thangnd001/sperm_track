import os
import pickle
import re
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import tensorflow_addons as tfa
# import torch
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# Analytics data
def circle_pie(labels, sizes, title='Analytics', size_image=[10, 10], legend='lower left'):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig1, ax1 = plt.subplots()
    plt.rcParams['figure.figsize'] = size_image
    ax1.pie(sizes, labels=labels, autopct='%1.2f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title(title)
    ax1.legend(loc=legend)

    plt.show()


def view_analytics_data_post(people, performance, x_label='Performance', title='Analytics',
                             size_image=[10, 10], legend='lower left'):
    plt.rcdefaults()
    plt.rcParams['figure.figsize'] = size_image
    fig, ax = plt.subplots()
    # Data
    y_pos = np.arange(len(people))
    error = np.random.rand(len(people))

    ax.barh(y_pos, performance, xerr=error, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(people)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc=legend)
    plt.show()


def view_data(global_labels_not_one_hot, size_image=[5, 15], legend='lower left', title='Analytics'):
    label_unique = np.unique(global_labels_not_one_hot)
    data_view = []
    for lb in label_unique:
        data_view.append(sum(global_labels_not_one_hot == lb))

    labels_plot = []
    data_plot = []
    max_name = "ZERO"
    max_count = 0
    cnt_u500 = 0
    for i in range(0, len(label_unique)):
        if max_count < data_view[i]:
            max_count = data_view[i]
            max_name = label_unique[i]
        if data_view[i] > 400:
            print(label_unique[i], ":", data_view[i])
            data_plot.append(data_view[i])
            labels_plot.append(label_unique[i])
        else:
            cnt_u500 += data_view[i]
    data_plot.append(cnt_u500)
    labels_plot.append('orther')

    print("BEST NAME:", max_name, " -  BEST COUNT: ", max_count)

    circle_pie(labels_plot, data_plot, size_image=size_image, legend=legend, title=title)
    view_analytics_data_post(labels_plot, data_plot, size_image=size_image, legend=legend, title=title)
    return np.array(label_unique), np.array(data_view)


def print_roc_avg_score(avg_score, name=""):
    print("===== ROC Avg", name, "Score =====")
    for i in range(0, len(avg_score)):
        print("ROC", i, ": %.4f" % avg_score[i])
    print("ROC AVG", name, ": %.4f" % np.average(avg_score))
    avg_score.append(np.average(avg_score))
    print("===============")
    dict_data = {
        name: avg_score
    }
    return dict_data


def set_gpu_limit(set_memory=5):
    memory_limit = set_memory * 1024
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def save_dump(file_path, data, labels):
    file = open(file_path, 'wb')
    # dump information to that file
    pickle.dump((data, labels), file)
    # close the file
    file.close()
    pass


def convert_gray(list_image):
    list_image = list_image.copy()
    results = []
    for image in list_image:
        img_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_cv = np.expand_dims(img_cv, axis=-1)
        results.append(img_cv)

    return np.array(results)

def load_data(path_file):
    file = open(path_file, 'rb')
    # dump information to that file
    (pixels, labels) = pickle.load(file)
    # close the file
    file.close()
    print(pixels.shape)
    print(labels.shape)
    print("TYPE DATA = ", type(pixels))
    
    # convert gray
    pixels = convert_gray(pixels)
    print(pixels.shape)
    return pixels, labels


def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
    axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1), len(model_history.history[acc]) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    # plt.savefig('roc.png')


def plot_model_legend(model_history):
    # view
    accuracy = model_history.history['accuracy']
    val_accuracy = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation')
    plt.xlim(0, len(accuracy))
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.show()


def print_cmx(y_true, y_pred, save_path="./folder_save", version="v-1.0"):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cmx, annot=True, fmt='g')
    print(cmx_data)
    # plt.show()
    plt.savefig(os.path.join(save_path, "result-cmx" + version + ".png"))


def get_callbacks_list(diractory,
                       status_tensorboard=True,
                       status_checkpoint=True,
                       status_earlystop=True,
                       file_ckpt="ghtk-spamham-weights-best-training-file.hdf5",
                       ckpt_monitor='val_accuracy',
                       ckpt_mode='max',
                       early_stop_monitor="val_accuracy",
                       early_stop_mode="max",
                       early_stop_patience=5):
    callbacks_list = []
    save_path = []
    if status_earlystop:
        # Early Stopping
        callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor=early_stop_monitor, patience=early_stop_patience,
                                                               restore_best_weights=True, mode=early_stop_mode, verbose=1)
        callbacks_list.append(callback_early_stop)

    # create checkpoint
    if status_checkpoint:
        checkpoint_path = os.path.join(diractory, "checkpt")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        # file="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        file_path = os.path.join(checkpoint_path, file_ckpt)
        save_path.append(file_path)

        checkpoint_callback = ModelCheckpoint(file_path, monitor=ckpt_monitor, verbose=1,
                                              save_best_only=True, mode=ckpt_mode, save_weights_only=True)
        callbacks_list.append(checkpoint_callback)

    # Tensorflow Board
    if status_tensorboard:
        tensorboard_path = os.path.join(diractory, "tensorboard-logs")
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
        callbacks_list.append(tensorboard_callback)
        save_path.append(tensorboard_path)

    return callbacks_list, save_path


def save_results_to_csv(results, directory="/home/quangdm/rnd/dataset/mail_spamham/results", name_bert="xlm-roberta-large", version="version-0.0"):
    dict_bert = {}
    for res in results:
        dict_bert.update(res)
    df_res = pd.DataFrame(data=dict_bert)
    file_name = name_bert + "-" + version + "-result.csv"
    df_res.to_csv(os.path.join(directory, file_name), encoding="utf-8")
    print("SAVE DONE")
    pass


def write_score(path="test.txt", mode_write="a", rows="STT", cols=[1.0, 2.0, 3.0]):
    file = open(path, mode_write)
    file.write(str(rows) + "*")
    for col in cols:
        file.write(str(col) + "*")
    file.write("\n")
    file.close()
    pass
