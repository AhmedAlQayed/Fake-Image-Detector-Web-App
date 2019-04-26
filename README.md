
# Fake Image Detector Web App



# Introduction

This repository provides you source code to train a deep convolutional neuron network ResNet50 based on a dataset from The IEEE Information Forensics and Security Technical Committee (IFS-TC) challenge to classify fake and pristine images.

The dataset used for training model is phase1 training dataset which can be downloaded from this link: http://ifc.recod.ic.unicamp.br/fc.website/index.py?sec=5

Our main focus is to:

1-Preprocess the dataset in order to have an effective training with mimimum amount of time.

2-Train a network using transfer learning technique to accelerate the process of getting an acceptable accuracy. 

The trained model was bases on ResNet50 pre-trained model, and has reached an accuracy of ~90%, you can check the python notebook "Training_model.ipynb" if you would like to investigate the possibility of increasing the accuracy.

After training the model, it can be used by a Flask application which allows the user to upload an image (Fake or Pristine) and have the model predict the image's label. If you would like to know more about how the web app based on Flask was developed, you can refer to the following link: https://www.mvmanh.com/machine-learning/huan-luyen-mo-hinh-deep-learning-de-phan-loai-anh-va-trien-khai-su-dung-trong-thuc-te.html.

We use Keras training our deep learning networking, an high-level deep learning library that offers a developer-friendly deep learning framework which is built on top of the tensorflow library.

# How to use

1-For Data Preprocessing and Training please refer to the jupyter notebook. Required dependencies are: *tensorflow*, *keras*, *numpy*, *scikit-learn*, *open-cv, and *python*.

2-For Deploying the app:
Please downloading our pre-trained model from this link:
https://www.dropbox.com/s/3qbu5n48gdnia3z/keras_model4.hdf5.zip?dl=0

Once you have downloaded the model model, please move it *****_static_***** folder of Flask app and run the following commands to start the app:

    export FLASK_APP=app.py
    python -m flask run


Using any web browser to access:  http://127.0.0.1:5000/
