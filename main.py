from flask import Flask,request,render_template,redirect
import os
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

covid_path  = './COVID/images/'
normal_path = './Normal/images/'

covid_files = [f for f in listdir(covid_path) if isfile(join(covid_path, f))]
normal_files = [f for f in listdir(normal_path) if isfile(join(normal_path, f))]

# load training cases for positive samples 

sum_pos = []
for i in range(300):
    file_name = covid_files[i]
    file_path = covid_path + file_name
    img = cv2.imread(file_path)
    img_new = cv2.resize(img, (32,32))
    img_array = np.array(img_new)
    img_array = img_array.ravel()
    sum_pos.append(img_array)
sum_pos = np.array(sum_pos)


# load training cases for negative samples 
sum_neg = []
for j in range(300):
    file_name = normal_files[j]
    file_path = normal_path + file_name
    img = cv2.imread(file_path)
    img_new = cv2.resize(img, (32,32))
    img_array = np.array(img_new)
    img_array = img_array.ravel()
    sum_neg.append(img_array)
sum_neg = np.array(sum_neg)

y_pos = np.ones((300,1))
y_neg = np.zeros((300,1))

X = np.concatenate((sum_pos,sum_neg))
y = np.concatenate((y_pos,y_neg))
y = y.ravel() 

X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify = y)

scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.fit_transform(X_train)


clf = svm.SVC(probability=True)
clf.fit(X_train_transformed, y_train)

def covid_test(test_img_path):
    test_img = cv2.imread(test_img_path)
    test_img_new = cv2.resize(test_img, (32,32))
    test_img_array = np.array(test_img_new)
    test_img_array = np.asarray(test_img_array.ravel())
    test_img_array = test_img_array/255.0
    test_img_array = np.array(test_img_array.reshape(1, -1))
    test_img_transformed = scaler.fit_transform(test_img_array)
    test_predict = clf.predict_proba(test_img_transformed)

    proba_pred = test_predict[:,1]
    if proba_pred > 0.5:
        res = 'Covid, '+ 'Covid probability' + str(proba_pred)
    else:
        res = 'Normal, '+ 'Covid probability' + str(proba_pred)
    
    return res



app = Flask(__name__)


# change the directory here to the directory of the Image folder under static folder
app.config["IMAGE_UPLOADS"] = "./static/Images"
#app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]

from werkzeug.utils import secure_filename

# create buttons to choose the image to upload and submit the upload image
@app.route('/home',methods = ["GET","POST"])
def upload_image():
	if request.method == "POST":
		image = request.files['file']
		if image.filename == '':
			print("Image must have a file name")
			return redirect(request.url)


		filename = secure_filename(image.filename)
		basedir = os.path.abspath(os.path.dirname(__file__))
		image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))
		return render_template("main.html",filename=filename,val=covid_test('./static/Images/'+ filename))

         

	return render_template('main.html')

# display the uploaded image 
@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static',filename = "/Images" + filename), code=301)



app.run(debug=False,port=2000)