from flask import Flask, request, jsonify, render_template, url_for
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler



# load the model and scaler from disk
model=pickle.load(open('model.pkl', 'rb'))
scaler=pickle.load(open('scaler.pkl', 'rb'))




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

'''
@app.route('/predict', methods=['GET','POST'])
def predict():
    return render_template('predict.html',prediction_text=5)
'''

@app.route('/predict', methods=['POST', 'GET'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features=scaler.transform(final_features)
    prediction = model.predict(final_features)
    
    
    if prediction[0]==1:
        return render_template('home.html', prediction_text="Good choice, your wine has been classified as good quality.")
    else:
        return render_template('home.html', prediction_text="Oh no, your wine has been classified as poor quality.")
        
@app.errorhandler(500)
def page_not_found(e):
    return "<h1>Oh no, something went wrong! Please check if your input values are valid numbers and try again!</h1>"



if __name__ == "__main__":
    app.run(debug=False)
    
    

