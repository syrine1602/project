import streamlit as st
import numpy as np
import cv2
import joblib
import cv2
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from PIL import Image

#Dict:
dic={
    "0":"George W. Bush",
    "1":"Alan Turing",
    "2":"Albert Einstein",
    "3":"Ons Jabeur",
    "4":"Oussama Mellouli"
}

#load models :
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("standard_scaler.pkl")
vgg_model = VGG16(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)
#Import haacascade model for face extraction 
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def get_embedding(model, face):

    face = face.astype("float32")

    face = np.expand_dims(face, axis=0)

    face = preprocess_input(face)

    embedding = model.predict(face, verbose=0)

    return embedding[0]

#Application layout :
st.title("Celebrity Face Recognition")

st.sidebar.subheader("Celebrity Face Recognition")
page = st.sidebar.radio(
    "",
    ["Home", "Recognition"]
)

if page == "Home":
    st.subheader("🏠 Home")
    st.write("Welcome to the Celebrity Recognition App!")
    st.write("""
        Upload a photo and discover which celebrity it matches.

        Available celebrities:
        - Albert Einstein
        - Alan Turing
        - George W. Bush
        - Ons Jabeur
        - Oussama Mellouli
        """)

    st.success("Ready? Go to the Recognition page and upload an image!")

else:
    st.subheader("👤 Recognition")

    st.write(
        "Upload a photo and the system will extract the face."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        
        st.image(
            image,
            use_container_width=True
        )

        if st.button("Predict celebrity",width="stretch"):

            image=np.array(image)
            faces = face_cascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(10,10)
            )
            if len(faces) == 0:
                st.error('no face detected')
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face,(224,224))
            if face is None:
                 st.error("No face detected!")
                 st.stop()
            else:
                
                image_embedding=get_embedding(vgg_model,face)
                image_embedding=np.expand_dims(image_embedding,axis=0)
                image_scaled=scaler.transform(image_embedding)
                st.write("You celebrity is :")
        
                prediction=svm_model.predict(image_scaled)
                label = prediction[0]

                st.success(dic[str(label)])

        