#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from datetime import datetime 
# Load the Sign Language MNIST dataset
df = pd.read_csv('C:/Users/Yadav/Desktop/sign_mnist_train.csv')

# Separate labels and features
labels = df['label']
images = df.drop('label', axis=1).values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape the features to 28x28 images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(26, activation='softmax')  # 26 classes for the letters A-Z
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Translation component
def translate_gesture_to_phrase(gesture):
    # Placeholder translation function
    # Replace with actual translation logic
    return chr(ord('A') + gesture)  # Assuming classes are labeled A-Z

# Time-based prediction restriction
def is_valid_prediction_time():
    current_time = datetime.now().time()
    return datetime.strptime("18:00", "%H:%M").time() <= current_time <= datetime.strptime("22:00", "%H:%M").time()

# Preprocess image function
def preprocess_image(image_path):
    # Open the image
    image = Image.open(image_path)
    # Resize the image to the required input size of the model
    image = image.resize((28, 28))
    # Convert the image to grayscale if needed
    image = image.convert('L')
    # Convert the image to a numpy array and normalize pixel values
    image = np.array(image) / 255.0
    # Expand dimensions to match the input shape expected by the model
    image = np.expand_dims(image, axis=-1)
    # Return the preprocessed image
    return image

# GUI setup
st.title("Sign Language Recognition")

uploaded_file = st.file_uploader("Upload a sign language image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = preprocess_image(uploaded_file)
        if is_valid_prediction_time():
            predicted_gesture = model.predict(image)
            predicted_phrase = translate_gesture_to_phrase(np.argmax(predicted_gesture))
            st.write("Predicted Phrase:", predicted_phrase)
        else:
            st.write("Sign language prediction is only available between 6 PM and 10 PM.")
    except Exception as e:
        st.error("An error occurred while processing the image: {}".format(e))

