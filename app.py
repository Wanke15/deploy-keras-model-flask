#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
#for matrix math
import numpy as np
import json

#for load our preprocessor(to use the preprocessor sklearn >= 0.20.0 is required)
from sklearn.externals import joblib
#arg of the preprocessor is a pandas DataFrame object
import pandas as pd

#for importing our keras model
import keras.models

#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import * 
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph, preprosessor
#initialize these variables
model, graph = init()
preprocessor = joblib.load("preprocessor.prep")

#decoding an image from base64 into raw representation
def preprocess(data):
	data = pd.DataFrame({"Type":[data["Type"]], "Group":[data["Group"]], "Priority":[data["Priority"]], 
						"Severity":[data["Severity"]], "Desc_vec":[data["Desc_vec"]]})
	preprocessed_data = preprocessor.transform(data).reshape(1, -1)
	return preprocessed_data
	
@app.route('/predict/',methods=['POST'])
def predict():
	#whenever the predict method is called, we're going
	#to perform inference, and return the prediction
	#get the raw data format
	postData = request.get_data()
	#jsonfy
	postData = json.loads(postData)
	X = preprocess(postData)
	with graph.as_default():
		cost_pred, duration_pred = model.predict(X)
		response = {"costPred":float('%.3f' % cost_pred[0][0]), "durationPred":float('%.3f' % duration_pred[0][0])}
		response = json.dumps(response)
		return response
	

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
