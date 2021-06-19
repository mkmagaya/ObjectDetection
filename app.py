# Importing required libraries,
import streamlit as st
import cv2
from PIL import Image
import glob
import numpy as np
import os
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.utils import plot_model
import glob
import matplotlib.pyplot as plt
from streamlit_player import st_player


# impoting pretrained and optimized model 
# model = load_model('vgg16Model.h5')

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
# students = st.markdown(html_side_temp, unsafe_allow_html = True)
st.sidebar.markdown(html_side_temp, unsafe_allow_html = True)


def main():
    videoLocation = st.file_uploader("Upload a Video:", type=["mp4"])
    btn = st.button("Detect and Predict")
    searchimage = st.text_input("Enter filename: ")
    searchbtn = st.button("Search Files")
    # st.sidebar.title("")
    temporary_location = False
    return (videoLocation)
    if videoLocation is not None:
        cap = cv2.VideoCapture(videoLocation)
        k=cap.isOpened()
        if k==False:
            video_file = open(videoLocation)
            video_bytes = video_file.read()
            st.video(video_bytes)
            if not os.path.exists('data'):
                os.makedirs('data')
            
        count = 0                                                                                                                                                                                                                                                                                                                                                                              
        while(True):
            ret, frame = cap.read()
            if not ret: 
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
            name = './kbsLogFiles/frames/' + 'raw' + str(count) + '.jpg'
            print('Framing Vieo ....' + name)
            cv2.imwrite(name, frame)
            count += 1 
    imagesLocation = "./kbsLogFiles/frames/*.jpg"
    images = []
    for filename in glob.glob(imagesLocation):
        imageFrame = image.load_img(filename, color_mode='rgb', target_size=(224,224)) 
        images.append(imageFrame)
    # iterating over the frames
    framedArr = []
    for imageItem in images:
    
        plt.imshow(imageItem)
        # transforming frames to array    
        imageArray = image.img_to_array(imageItem)

        # expanding dimensions
        imageArray = np.expand_dims(imageArray, axis = 0)
            
        # preprocessing data    
        imageArray = preprocess_input(imageArray)
            
        # detecting and predicting from frames
        if btn:
            # result = prediction()
            features = model.predict(imageArray)
            # Embed a youtube video
            st_player(videoLocation)
            # def record_data(frame):
            cv2.imwrite("./filedata/", features)
        elif searchbtn:
            searchFor(searchimage)
            output(searchResult)


                   
        #     # decoding predictions  
        #     result = decode_predictions(features)
        # framedArr.append(result)                    





if __name__=='__main__':
	main()




