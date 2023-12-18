import pickle
import cv2
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score
from scripts.sperm_classification import SpermClassification



CATEGORIES = ['Normal', 'Tapered', 'Pyriform', 'Amorphous']
CATEGORIES = ['01_Normal', '02_Tapered', '03_Pyriform', '04_Amorphous']


# def load_data(path_file):
#     file = open(path_file, 'rb')
#     # dump information to that file
#     (pixels, labels) = pickle.load(file)
#     # close the file
#     file.close()
#     print(pixels.shape)
#     print(labels.shape)
#     return pixels, labels

def load_data(path_file):
    file = open(path_file, 'rb')
    # dump information to that file
    (pixels, labels) = pickle.load(file)
    # close the file
    file.close()
    print(pixels.shape)
    print(labels.shape)
    print("TYPE DATA = ", type(pixels))
    return pixels, labels

def main(data_dir: str, model_path: str, device: str):
    input_size = 40
    pixels, labels = load_data(data_dir)
    classificator = SpermClassification(model_path, device, input_size)

    #
    lb = preprocessing.LabelBinarizer()
    labels_train_one_hot = lb.fit_transform(labels)
    y_true = np.argmax(labels_train_one_hot, axis=1)
    # print("LABEL = ", labels)
    # print("Y TRUE = ", y_true)
    
    # y_target = np.argmax(y_predict, axis=1)


    # list data
    list_data = []
    for i, label in enumerate(labels):
        list_data.append(pixels[i])
        cv2.imwrite(f'check_dataset/{i}_{label.split("_")[-1]}.png', np.array(pixels[i], dtype=np.uint8))
    
    # predict
    results = []
    result_one_hot = []
    preds = classificator(list_data, batch_size=16)
    for pred in preds:
        results.append(CATEGORIES[pred])
        result_one_hot.append(pred)

    # check result
    # counter = 0
    # check_result = []
    # print(results)
    # for pred, target in zip(results, labels):
    #     check_result.append(f'{pred}=>{target}')
    #     print(f"COMPARE = {pred} == {target} => {pred == target}")
    #     if pred == target:
            # counter += 1

    # print("CHECK RESULT = ", check_result)
    print('COUNTER = ', len(preds))
    # print(f"Accuracy = {100*counter/len(preds)} \%")
    print(f'Accuracy = {accuracy_score(y_true, np.array(result_one_hot))}')


if __name__ == '__main__':
    data_dir = 'data/HuSHeM_datatest.data'
    # model_path = 'sperm_classification/model/sperm/hushem-version-0.1-weights-best-k-fold-2.h5'
    model_path = 'sperm_classification/runs/training/HuSHeM_dataset/version-0.0/model-save/model-HuSHeM_dataset-version-0.0.h5'
    device = '/device:GPU:0'
    main(data_dir, model_path, device)




    