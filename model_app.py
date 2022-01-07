import cv2
import os
import pickle
import cv2
import os.path
import tensorflow
import numpy as np
import streamlit as st
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

#converts an image to encoding by passing it through our trained model
def img_to_encoding(image, model):
    img1 = cv2.resize(image, (160,160))
    x_train = np.array([img1])
    embedding = model.predict_on_batch(x_train)
    return embedding

input_shape = (3, 160, 160)
faces = []
images = {}
FRmodel = load_model('facenet_keras.h5')

# initialize the user database
def ini_user_database():
    # check for existing database
    if os.path.exists('database/user_dict.pickle'):
        with open('database/user_dict.pickle', 'rb') as handle:
            user_db = pickle.load(handle)   
    else:
        # make a new one
        # we use a dict for keeping track of mapping of each person with his/her face encoding
        user_db = {}
        # create the directory for saving the db pickle file
        os.makedirs('database')
        with open('database/user_dict.pickle', 'wb') as handle:
            pickle.dump(user_db, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    return user_db

# we use a dict for keeping track of mapping of each person with his/her face encoding
user_db = ini_user_database()


st.set_page_config(page_title="Find'em", page_icon="🖖")
#Condense the layout
padding = 1
st.markdown(f""" 
<style>
.reportview-container .main .block-container{{
    padding-top: {padding}rem;
    padding-right: {padding}rem;
    padding-left: {padding}rem;
    padding-bottom: {padding}rem;
}} </style> """, unsafe_allow_html=True)


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

def sample():
    html_temp_sample = """
    <body>
    <br>
    <div>
    <ul>    
    <li><p>This is page is currently work in progress!!</p></li>
    <li><p>It is a simple interface for the deployment of the facial recognition model. 
    Users are able to upload the images via the file uploader. The image is then passed to the model when the user clicks on the "Recognize"button.</p></li>
    <li><p>The results of the prediction will then be displayed on the page; regarding as to whether or not a match was found.</p></li>
    </ul>
    </div>
    </body>
    """
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(html_temp_sample, unsafe_allow_html=True)
    st.file_uploader("Upload an image to find match in the database.", type=['jpg', 'png','jpeg'], key=None, accept_multiple_files=False)
    st.button("Recognize")

def home():
    """
    The face recognition homepage
    """
    html_temp_home = """
    <body>
    <br>
    <div>    
    <p>Facial Recognition is a technology capable of matching a human face from a digital image or a video 
    frame against a database of faces, typically employed to authenticate users through ID verification services, 
    works by pinpointing and measuring facial features from a given image.</p>
    <h5 style="text-decoration:underline;">Facial Recogntion Pipeline:</h5>
    <ul>
    <li><h5>Face Detection:</h5><p>A Multi-Task Convolutional Neural Network is used to do face detection.</p></li>
    <li><h5>Face Cropping:</h5><p>Cropping of faces from images</p></li>
    <li><h5>Face Alignment:</h5><p>Align faces by the eye line</p></li>
    <li><h5>Face Encoding:</h5><p>Extract encoding from the face using our trained model.</p></li>
    <li><h5>Face Classification:</h5><p>Find similarity between the faces via euclidean distances between face encodings</p></li>
    </ul> 
    <p>Use the sidebar to choose what you would like to do.</p>
    </div>
    </body>
    """
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(html_temp_home, unsafe_allow_html=True)

def upload_person():    
    """
    The face recognition app
    """
    html_temp_upload = """
    <div>    
    <p>Did you come across a missing person?<br>
    On this page, you can upload a facial image of the missing person to the database.</p>
    </div>
    </body>
    """
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(html_temp_upload, unsafe_allow_html=True)

    # adds a new user face to the database using his/her image stored on disk using the image path
    def add_user_img(user_db, FRmodel, name, image):
        if name not in user_db: 
            user_db[name] = img_to_encoding(image, FRmodel)
            # save the database
            with open('database/user_dict.pickle', 'wb') as handle:
                    pickle.dump(user_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
            st.success('User ' + name + ' added successfully')
        else:
            st.warning(f'The name {name} is already registered! Try a different name.........')
    
    # deletes a registered user from database
    def delete_user(user_db, name):
        popped = user_db.pop(name, None)
        
        if popped is not None:
            st.success('User ' + name + ' deleted successfully')
            # save the database
            with open('database/user_dict.pickle', 'wb') as handle:
                    pickle.dump(user_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif popped == None:
            st.warning('No such user !!')

    #form to upload a missing person to the database
    with st.form(key = "form1"):
        name = st.text_input(label = "Enter the name of the person (*required)")
        upload_image = st.file_uploader("Upload an image to the database.", type=['jpg', 'png','jpeg'], key=None, accept_multiple_files=False)
        if st.form_submit_button(label = "Upload"):
            if not name:
                st.warning("Please fill out the required fields")
            else:           
                with st.spinner(text='Processing............'):
                    if upload_image is not None:
                        image = Image.open(upload_image)
                        img_array = np.array(image.convert('RGB'))
                        src = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                        detector = MTCNN()
                        bboxes = detector.detect_faces(img) 
                        if bboxes == []:
                            st.warning("Face not detected! Please upload an image with a visible face")
                        else:
                            st.success("Face successfully detected")
                            # add a user
                            add_user_img(user_db, FRmodel, name, img)

        #form to delete a user from the database
        # with st.form(key = "form2"):
        #     name2 = st.text_input(label = "Enter the name of the person to delete")
        #     if st.form_submit_button(label = "Delete user"):
        #         delete_user(user_db, name2)


def find_person():    
    """
    The face recognition app
    """
    html_temp_find = """
    <body>
    <div>    
    <p>The model used here has been trained on the <a href="http://vis-www.cs.umass.edu/lfw/" target="_blank">Labelled Faces in the World (LFW)</a> dataset. 
    To test it, you can upload images of people like: Arnold_Schwarzenegger, Bill_Gates, Charles Moose, George Bush, 
    Tom_Ridge e.t.c. to the database for positive results
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

    # recognize the input user face encoding by checking for it in the database
    def find_face(image, database, model, threshold):
        # find the face encodings for the input image
        encoding = img_to_encoding(image, model)
        
        min_dist = 99999
        # loop over all the recorded encodings in database 
        for name in database:
            # find the similarity between the input encodings and claimed person's encodings using L2 norm
            dist = np.linalg.norm(np.subtract(database[name], encoding) )
            # check if minimum distance or not
            if dist < min_dist:
                min_dist = dist
                identity = name
        
        if (min_dist*0.1) > threshold: 
            st.warning("Person not found in the database. Consider uploading their facial image in the Upload Person rab.")
        else:
            st.success(f"Match found! Person identified as {str(identity)}")


    find_image = st.file_uploader("Upload an image to find match in the database.", type=['jpg', 'png','jpeg'], key=None, accept_multiple_files=False)
    if find_image is not None:
        if st.button("Search"):
            with st.spinner(text='Processing............'):
                if find_image is not None:
                    image = Image.open(find_image)
                    img_array = np.array(image.convert('RGB'))
                    src = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                    detector = MTCNN()
                    bboxes = detector.detect_faces(img) 
                    if bboxes == []:
                        st.warning("Face not detected! Please upload an image with a visible face")
                    else:
                        st.success("Face successfully detected")
                        find_face(img, user_db, FRmodel, threshold=0.4)
                    
    

def main():
    html_temp_main = """
    <body>
    <div class="title">
    <h2>Using Facial Recognition to identify missing persons</h2>
    </div>
    </body>
    """
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after{
	content:'Made by kencodest';
	visibility: visible;
	display: block;
	position: relative;
	padding: 5px;
	top: 2px;
    }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(html_temp_main, unsafe_allow_html=True) 

    choice = st.sidebar.selectbox("What do you want to do?", ('Home', 'Upload Person', 'Find Person'))
    if choice == 'Home':
        home()

    elif choice == 'Upload Person':
        upload_person()

    elif choice == 'Find Person':
        find_person()

    else:
        home()
    

if __name__ == '__main__':
    main()
