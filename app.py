from flask import Flask, request, jsonify
import pickle
from collections.abc import Mapping
import numpy as np

model = pickle.load(open('linearmodel.pkl', 'rb'))

app = Flask(__name__)


@app.route('/predictApi', methods=['POST'])
def predict():
    data1 = request.json['bedrooms']
    data2 = request.json['bathrooms']
    data3 = request.json['sqft_living']
    data4 = request.json['sqft_lot']
    data5 = request.json['floors']
    data6 = request.json['view']
    data7 = request.json['condition']
    data8 = request.json['grade']
    data9 = request.json['sqft_basement']
    data10 = request.json['sqft_living15']
    data11 = request.json['sqft_lot15']
    data12 = request.json['yr_built_binned_1']
    data13 = request.json['yr_built_binned_2']
    data14 = request.json['yr_built_binned_3']
    data15 = request.json['yr_built_binned_4']
    data16= request.json['yr_built_binned_5']
    data17 = request.json['yr_built_binned_6']
    data18 = request.json['yr_built_binned_7']
    data19 = request.json['yr_built_binned_8']


    arr = np.array([[data1, data2, data3, data4, data5,
                     data6, data7, data8, data9,data10, data11, data12, data13, data14, data15,
                     data16, data17, data18, data19]])
    

    prediction = model.predict(np.array(arr).tolist()).tolist()
    return jsonify({'prediction': prediction})
if __name__ == "__main__":
    app.run(debug=True) 