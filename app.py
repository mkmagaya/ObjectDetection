# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:58:54 2021

@author: Admin
"""

from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from objectdetection import ObjectDetection as ObjD
import os

UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DO =  ObjD()
@app.route('/', methods=["GET"])
def index():
    return render_template('home.html')

@app.route('/', methods=["POST"])
def upload():
    if request.method == 'POST':
        file = request.files['videoInput']
        filename = secure_filename(file.filename)
        file_path =os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        DO.to_frames(file_path)
        DO.detect_object()
        return redirect('/')
   
@app.route('/search', methods=['GET', 'POST'])
def search():
    search_text = request.form.get("searchInput")
    objects = DO.search_for(search_text)
    return render_template('search.html', frames_list=objects, type_list=type(objects)==list, search_txt=search_text)

if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)

