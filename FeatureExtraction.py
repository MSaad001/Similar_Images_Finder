import tensorflow
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import cv2
import numpy as np
from numpy.linalg import norm
# import os
# from tqdm import tqdm
# import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.summary()

img = cv2.imread("123.jpg")
img = cv2.resize(img, (224, 224))
img = np.array(img)
# print(img.shape)

expand_img = np.expand_dims(img, axis=0)
# print(expand_img.shape)

pre_img = preprocess_input(expand_img)
# print(pre_img.shape)

result = model.predict(pre_img).flatten()
# print(result.shape)

normalized_result = result / norm(result)


# print(normalized_result.shape)

def extract_features(img_path, model1):
    img1 = cv2.imread(img_path)
    img1 = cv2.resize(img1, (224, 224))
    img1 = np.array(img1)
    expand_img1 = np.expand_dims(img1, axis=0)
    pre_img1 = preprocess_input(expand_img1)
    result1 = model1.predict(pre_img1).flatten()
    normalized_result1 = result1 / norm(result1)
    return normalized_result1


# print(extract_features("1533.jpg", model))

filename = []
feature_list = []

# for file in os.listdir('Dataset'):
#     filename.append(os.path.join('Dataset', file))
# for file in tqdm(filename):
#     feature_list.append(extract_features(file, model))
# pickle.dump(filename, open('filenames.pkl', 'wb'))
# pickle.dump(feature_list, open('featureVector.pkl', 'wb'))
