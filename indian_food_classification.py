import streamlit as st
import numpy as np
from tensorflow.image import resize
from tensorflow.keras.models import load_model
from PIL import Image

st.title("Indian Food Classification 😋🍔🍗")
img = st.file_uploader("Upload an image")

if "image_label" not in st.session_state:
    st.session_state["image_label"] = False
if "food_name" not in st.session_state:
    st.session_state["food_name"] = ""

if img:
    img = Image.open(img)
    img = resize(img, size=(224, 224))
    img_show = np.array(img/255.)
    st.image(img_show, caption="Uploaded food image", width=224)
    img = np.expand_dims(np.array(img), axis=0)
    st.session_state["image_label"] = True


@st.cache_resource
def load_my_model():
    return load_model("saved_models\effnet_v2_b2")


@st.cache_resource()
def predict_image(img):
    food_labels = ['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'idli',
                   'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa']
    index = np.argmax(model.predict(img), axis=1)
    food = " ".join(food_labels[np.squeeze(index)].split("_")).capitalize()
    return food
    # st.markdown(f"### Predicted as :blue[{food}]")


if st.session_state["image_label"]:
    if st.button("Predict"):
        model = load_my_model()
        with st.spinner("Predicting..."):
            st.session_state["food_name"] = predict_image(img)


# To show a like button when the model predicted something
if st.session_state["food_name"] != "":
    st.markdown(f"### Predicted as :blue[{st.session_state['food_name']}]")
    like_btn = st.empty()
    if like_btn.button("👍", help="Is the prediction correct?"):
        st.balloons()
        like_btn.empty()
