import streamlit as st
import numpy as np
from PIL import Image
import json
import tensorflow as tf

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path='crop_disease_model.tflite')
    interpreter.allocate_tensors()
    with open('class_names.json') as f:
        class_names = json.load(f)
    return interpreter, class_names

interpreter, class_names = load_model()

def format_name(name):
    return name.replace('_', ' ').replace('  ', ' ')

def predict(img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

st.title("Crop Disease Detector")
st.write("Upload a leaf image to detect the disease")

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Uploaded image", use_column_width=True)

    preds = predict(img)
    top3_idx = np.argsort(preds)[::-1][:3]

    st.subheader("Results")
    for i, idx in enumerate(top3_idx):
        confidence = float(preds[idx]) * 100
        label = format_name(class_names[idx])
        if i == 0:
            st.success(f"Prediction: {label} ({confidence:.1f}%)")
        else:
            st.info(f"#{i+1}: {label} ({confidence:.1f}%)")