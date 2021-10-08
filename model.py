import cv2
import time
import streamlit as st
import numpy as np
from PIL import Image
from face_recognition_1 import FaceRecognition



STYLE = """
<style>


.title {
    background-color: teal;
}

h2 {
    text-transform: uppercase;
    color: white; 
    text-align: center;
}

h5 {
    color: teal;
}

p {
    font-size: 113%;
    line-height= 1.5em;
}
</style>
"""


def home():
    
    """
    The face recognition homepage
    """
    html_temp_home = """
    <body>
    <br>
    <div>    
    <p>This is a simple deployment of a facial recognition model trained using MTCNN and FaceNet.</p>
    <h5 style="text-decoration:underline;">Facial Recogntion Pipeline:</h5>
    <ul>
    <li><h5>Face Detection:</h5><p>The MTCNN() algorithm is used to do face detection. It can be found <a href="https://github.com/ipazc/mtcnn" target="_blank">here</a></p></li>
    <li><h5>Face Alignment:</h5><p>Align faces by the eye line</p></li>
    <li><h5>Face Encoding:</h5><p>Extract encoding from the face using FaceNet. The implementation of FaceNet can be found <a href="https://github.com/faustomorales/keras-facenet" target="_blank">here</a></p></li>
    <li><h5>Face Classification:</h5><p>Classify the faces via euclidean distances between face encodings</p></li>
    </ul>  
    <p>Use the left sidebar to choose what you would like to do.</p>
    </div>
    </body>
    """
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(html_temp_home, unsafe_allow_html=True)


def find_person():    
    """
    The face recognition app
    """
    html_temp_find = """
    <body>
    <div>    
    <p>The model used here has been trained on the <a href="http://vis-www.cs.umass.edu/lfw/" target="_blank">Labelled Faces in the World (LFW)</a> dataset with 90% accuracy. To test it, you can use
    images of people like: Arnold_Schwarzenegger, Bill_Gates, Charles Moose, George Bush, Tom_Ridge e.t.c. for positive results
    <br>
    <br>
    Nonetheless, any image can be used to test the model.
    </p>
    </div>
    <script language="javascript">
    document.querySelector("h1").style.color = "red";
    console.log("Streamlit runs JavaScript");
    alert("Streamlit runs JavaScript");
    </script>
    </body>
    """

    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(html_temp_find, unsafe_allow_html=True)  
  

    uploaded_image = st.file_uploader("Upload an image to find match in the database.", type=['jpg', 'png','jpeg'], key=None, accept_multiple_files=False)
    # person = "madonna"
    if uploaded_image is not None:
        if st.button("Recognize"):
            try:
                model = FaceRecognition()
                model.load("lfw_model.pkl")
                image = Image.open(uploaded_image)
                result = model.predict(image, threshold=0.6)
                if result["predictions"][0]["person"] == "UNKNOWN":
                    st.warning('No match found in database!!')
                else:
                    st.success(f'Match found! This person has been identified as {result["predictions"][0]["person"]}. Confidence: {round(100*result["predictions"][0]["confidence"], 2)}%.')
            except IndexError:
                st.warning(f'Face not detected! Recapture image and ensure that the face is fully visible')


def upload_person():    
    """
    The face recognition app
    """
    html_temp_upload = """
    <div>    
    <p>This page is currently under maintenance. Check back later.</p>
    </div>
    </body>
    """
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(html_temp_upload, unsafe_allow_html=True) 



def main():
    html_temp_main = """
    <body>
    <div class="title">
    <h2>Using Facial Recognition to identify missing persons</h2>
    </div>
    </body>
    """
    
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(html_temp_main, unsafe_allow_html=True) 

    choice = st.sidebar.selectbox("What do you want to do?", ('Home', 'Find Person', 'Upload Person'))
    if choice == 'Home':
        home()

    elif choice == 'Find Person':
        find_person()

    elif choice == 'Upload Person':
        upload_person()

    else:
        home()
    

if __name__ == '__main__':
    main()


