import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

MODEL_PATH = "Model/32x3-CNN.keras"
HOME_IMAGE = "Images/Banner.jpg"
CATEGORIES = ['Dog', 'Cat']

st.set_page_config(
    page_title="Cats and Dogs Classifier",
    page_icon="🐈",
    layout='wide',
    menu_items={
        'Get Help': 'https://github.com/Nasif-Azam',
        'Report a bug': 'mailto:nasifazam07@gmail.com',
        'About': "### Cats and Dogs Classifier\nThis app predicts cats and dogs using AI."
    }
)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Page:", ["Home", "About Project", "Classifier"])


def HomePage():
    st.header("Cats and Dogs Classifier")
    st.image(HOME_IMAGE)
    st.markdown(
        """
        **Using the CNN to build the classification system**        
        ### About Me
        **<i class="fa fa-user"></i> Name:** Nasif Azam\n
        **<i class="fa fa-envelope"></i> Email:** nasifazam07@gmail.com\n
        **<i class="fa fa-mobile-alt"></i> Phone:** +880-1533903305\n
        **<i class="fa fa-map-marker-alt"></i> Address:** Mirpur, Dhaka, Bangladesh

        <a href="https://www.facebook.com/md.nasif850" target="_blank" style="margin-right: 10px;"><i class="fab fa-facebook fa-2x"></i></a>
        <a href="https://github.com/Nasif-Azam" target="_blank" style="margin-right: 10px; color:black;"><i class="fab fa-github fa-2x"></i></a> 
        <a href="https://www.linkedin.com/in/nasif-azam-9aa2331a0/" target="_blank" style="margin-right: 10px; color:sky;"><i class="fab fa-linkedin fa-2x"></i></a> 
        <a href="https://www.hackerrank.com/profile/Nasif_Azam" target="_blank" style="margin-right: 10px; color:green;"><i class="fab fa-hackerrank fa-2x"></i></a> 
        <a href="https://www.kaggle.com/nasifazam" target="_blank" style="margin-right: 10px; color:blue;"><i class="fab fa-kaggle fa-2x"></i></a>  

        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """, unsafe_allow_html=True
    )


def AboutPage():
    st.header("About Project")
    st.markdown("""
    ### Dataset
    The Dogs vs. Cats dataset is a standard computer vision dataset that involves classifying photos as either containing a dog or cat.
    This dataset is provided as a subset of photos from a much larger dataset of 3 million manually annotated photos.
    The dataset was developed as a partnership between Petfinder.com and Microsoft.
    ### Content
    Download Size: 824 MB
    The data-set follows the following structure:\n
    \t|--> kagglecatsanddogs_3367a\n
    \t\t| |--> readme[1].txt\n
    \t\t| |--> MSR-LA - 3467.docx\n
    \t\t| |--> PetImages\n
    \t\t\t| | |--- Cat (Contains 12491 images)\n
    \t\t\t| | |--- Dog (Contains 12470 images)
    Needed the **PetImages** directory
    ### Acknowledgements
    This data-set has been downloaded from the official Microsoft website: this link.    
     """, unsafe_allow_html=True)


def ClassifierPage():
    st.header("Prediction")
    test_image = st.file_uploader("Choose an Image: ", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, width=280, use_column_width=False)

        if st.button("Predict"):
            st.balloons()
            result = Model_Prediction(image)
            st.write("Model Predicting... ")
            st.success(result)
    else:
        st.markdown('<p style="color:red;">Upload An Image (.jpg, .jpeg, .png) First!!</p>', unsafe_allow_html=True)


def Model_Prediction(image):
    IMG_SIZE = 50
    image = np.array(image.convert('L'))  # Convert to grayscale
    img_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    model = tf.keras.models.load_model(MODEL_PATH)
    prediction = model.predict([img_array])

    label = CATEGORIES[int(prediction[0][0])]
    return label


if app_mode == "Home":
    HomePage()
elif app_mode == "About Project":
    AboutPage()
else:
    ClassifierPage()
