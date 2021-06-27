import numpy as np
import os
import cv2
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input, decode_predictions
from glob import glob
from PIL import Image
import streamlit as st

# importing pretrained and optimized model 
model = load_model('vgg16Model.h5')

# OpenCV and VGG16 object detection 
class ObjectDetection():
    
    def __init__(self):
        self.objects = []

    # function to break videos into frames
    def to_frames(self, video_upload):
        cap = cv2.VideoCapture(video_upload.name)
        if not os.path.exists('data'):
            os.makedirs('data')
        count = 0
        while(True):
          ret, frame = cap.read()
          if not ret: 
                break
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
          name = 'static/frames/' + 'raw' + str(count) + '.jpg'
          print('Framing Video ....' + name)
          cv2.imwrite(name, frame)
          count += 1
    
    # detecting objects from frames
    def detect_object(self):
        for item in self.get_frames():  
            original_image = load_img(item, target_size=(224, 224))
            image = img_to_array(original_image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            y_pred = model.predict(image)
            label = decode_predictions(y_pred)
            self.objects.append(label[0][1][1])
            # st.write(image)                                                                                                                                                                                                                                                                   
        with open('detected_objects.txt', 'w') as f:
            f.write(json.dumps(self.objects))
                
    # capturing detected objects
    def get_objects(self):
        return self.objects
    
    # setting frames into an array to be searchable
    def get_frames(self):
        FRAMES_ARR =  glob("static/frames/*.jpg")
        return FRAMES_ARR
    
    # function to search for detected objects
    def search_for(self, _object):
        with open('detected_objects.txt', 'r') as f:
            objects = list(json.loads(f.read()))
        results = []
        if _object in set(objects):
            for i in range(len(objects)):
                if _object.__eq__(objects[i]):
                    frame_path = self.get_frames()[i]
                    frame_path = frame_path.split('\\')[1]
                    results.append(frame_path)                
        else:
            return "There is no such file"
        return results

# creating an instance of the Object Detection class
detector=ObjectDetection()

# front ent rendering using streamlit
html_render_title = """
    <style="color:black;text-align:center;>KBS Assignment 2</style>
    """ 
html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:left;">Keras & OpenCV Project</h1>
    </div>
    """
html_side_temp = """
    <div style ="padding:13px">
    <h1 style ="color:black;text-align:center;">Project Engineers</h1>
    <h3 style ="color:black;text-align:center;">Magaya Makomborero r181571b</h3>
    <h3 style ="color:black;text-align:center;">Mabhuka Oswell r181573f</h3>
    </div>
    """    
st.title("KBS Assignment 2")
st.markdown(html_temp, unsafe_allow_html = True)
st.sidebar.markdown(html_side_temp, unsafe_allow_html = True)
x = st.slider('Change Threshold value',min_value = 50,max_value = 255)

videoLocation = st.file_uploader("Upload a Video:", type=["mp4"])
btn = st.button("Detect and Predict")

if btn:
    detector.to_frames(videoLocation)
    detector.detect_object()
    frames = detector.get_objects()
    st.write('Frames for', videoLocation.name)
    st.write(detector.get_objects())
       
searchimage = st.text_input("Enter Object Name: ")
searchbtn = st.button("Search Object")

if searchbtn:
    searched = detector.search_for(searchimage)
    searched_image = detector.search_for(searchimage)[0]
    st.write('Objects of', searchimage, 'detected in', videoLocation.name)
    st.write(searched)
    st.write(searched_image, 'represents the', searchimage, 'objects detected in', videoLocation.name)
    # original = Image.open(searched_image)
    # st.image(original, use_column_width=True)
    image_search = cv2.imread(searched[0])
    st.image(image_search, caption=searched[0])






