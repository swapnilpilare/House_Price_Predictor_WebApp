import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    features_array = np.array(features)
    reshaped_array = np.reshape(features_array,(1,-1))
    my_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('std_scaler', StandardScaler())])
    final_features = my_pipeline.fit_transform(reshaped_array.T)
    prediction = model.predict(final_features.T)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='House price should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)