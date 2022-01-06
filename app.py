# -*- coding: utf-8 -*-
import shap
import joblib
from flask import Flask, request, jsonify

#==============================================================================

model = joblib.load(open('input/home_credit_model.sav', 'rb'))
data_set = joblib.load(open('input/home_credit_data_test.sav', 'rb'))
id_list = data_set.index

#==============================================================================

app = Flask(__name__)

@app.route('/api', methods=['GET'])
def function_api():
    id_request = request.args.get('id')
    try:    
        id_client = int(id_request)
    except ValueError:
        id_client = -1
    
    if id_client not in id_list:
        message = "<h2>Error, request doesn't valid</h2>"
        return message
        
    client=data_set.loc[[id_client]]
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(client)
    shap_map = zip(client.columns, shap_values[0][0])  
    
    pred = model.predict(client)
    proba = model.predict_proba(client)
    
    return jsonify({
                    str(int(client.index.values)) : dict(shap_map),
                    'prediction': int(pred),
                    'proba': list(proba[0]),
                    })

@app.route('/gfi', methods=['GET'])
def function_ft_import():
    importances = model.feature_importances_
    importances_int = []
    for v in importances:
        importances_int.append(int(v))

    importances_map = zip(data_set.columns,importances_int)
    
    return jsonify({'global_feature_importance':dict(importances_map)})


@app.route('/id_list', methods=['GET'])
def function_list():
    id_list_int = []
    for i in id_list:
        id_list_int.append(int(i))
    
    return jsonify(id_list_int)

@app.route('/', methods=['GET'])
def home():    
    return "<h1>OC-P7-JB-API</h1>"

#==============================================================================

if __name__ == "__main__":
  app.run(host='127.0.0.1', port=5000, debug=True)