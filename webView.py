import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

feature_list = np.array(pickle.load(open('featureVector.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Similar Images Finder')


def save_uploaded_file(uploaded_file1):
    try:
        with open(os.path.join('uploads', uploaded_file1.name), 'wb') as f:
            f.write(uploaded_file1.getbuffer())
        return 1
    except:
        return 0


def extract_features(img_path, model1):
    img1 = cv2.imread(img_path)
    img1 = cv2.resize(img1, (224, 224))
    img1 = np.array(img1)
    expand_img1 = np.expand_dims(img1, axis=0)
    pre_img1 = preprocess_input(expand_img1)
    result1 = model1.predict(pre_img1).flatten()
    normalized_result1 = result1 / norm(result1)
    return normalized_result1


def recommend(features1, feature_list1):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list1)

    distances, indices1 = neighbors.kneighbors([features1])

    return indices1


# steps
# file upload => save
uploaded_file = st.file_uploader("Choose an Image")
print(uploaded_file)
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file on screen
        display_image = Image.open(uploaded_file)
        resized_img = display_image.resize((200, 200))
        st.image(resized_img)
        # feature extraction
        features = extract_features(os.path.join("uploads", uploaded_file.name), model)
        # st.text(features)
        # recommendation
        indices = recommend(features, feature_list)
        # show
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
        with col6:
            st.image(filenames[indices[0][5]])
    else:
        st.header("Upload Failed")