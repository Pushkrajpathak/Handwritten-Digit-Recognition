import streamlit as st
import tensorflow as tf
import cv2
from tensorflow import keras
import numpy as np
from PIL import Image
st.imwrite("trial")
# model_path = r"C:\Users\Pushkaraj\Downloads\Handwritten_digit_model.h5"
# # Load the saved Keras model
# model = keras.models.load_model(model_path, compile=False)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# def preprocess_image(image):
#     image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
#     # image=cv2.imread(uploaded_file,cv2.IMREAD_GRAYSCALE)
#     _, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
#     resized_image = cv2.resize(thresholded_image, (28, 28))
#     normalized_image = resized_image.astype('float32') / 255.0
#     preprocessed_image = np.expand_dims(normalized_image, axis=-1)
#     x1 = preprocessed_image.reshape(-1, 28, 28, 1)
#     return x1


# # Streamlit app code
# st.title("Digit Recognition")
# st.write("Upload an image of a digit (0-9) and the model will predict it.")

# uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
# if uploaded_file is not None:
#     # Read the image file
#     temp=uploaded_file
#     image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = preprocess_image(image)

#     predictions = np.argmax(model.predict(image))

#     # Display the uploaded image and the predicted digit

#     #st.image(temp, caption='Uploaded Image', use_column_width=True, width=0)
#     col1, col2 = st.columns([1, 2])
#     col1.image(temp, caption='Uploaded Image', use_column_width=True)
#     with col2:
#         st.subheader("Predicted Digit:")

#         st.write("The digit in the uploaded image:", predictions)
