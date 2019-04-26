import os
from flask import Flask, render_template, request
from flask import send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import math
from keras.preprocessing import image
import tensorflow as tf
from sklearn.feature_extraction import image as ims

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

graph = tf.get_default_graph()
with graph.as_default():
    # load model at very first
   model = load_model(STATIC_FOLDER + '/' + 'keras_model4.hdf5')
   model._make_predict_function()
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def apic(full_path):
    img =cv2.imread(full_path)
    img = cv2.resize(img,(350,350))
    nPatches=math.floor(img.size/(299*299))
    patches = ims.extract_patches_2d(img, (299, 299))
    threshold=0.1
    ratio=0
    print(nPatches)
    predictions=[]
    i=0
    print(len(patches))
    while (i<len(patches)):
        x = image.img_to_array(patches[i])
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predicted = model.predict(x)
        predictions.append(str(np.argmax(predicted)))
        i=i+300
    s=0
    j=0
    m=0
    while (s<len(predictions)):
        if (predictions[s])=='0':
            j=j+1
        if (predictions[s])=='1':
            m=m+1
        s=s+1
    if j>m:
        result=0
        if m>0:
            ratio=min(nPatches/j,1)
        else:
            ratio=1
    else:
        result=1
        ratio=0
    print(j)
    print(m)
    return result,ratio




# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        indices = {0: 'Fake', 1: 'Real'}
        result, ratio = apic(full_name)
        predicted_class=int(result)
        label = indices[predicted_class]
        accuracy = math.ceil(ratio*100)

    return render_template('predict.html', image_file_name = file.filename,label=label,accuracy=accuracy)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True
