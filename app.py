from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import XGBoostModel

app=Flask(__name__)
api=Api(app)

model = XGBoostModel()

clf_path = '/Users/shwetasaloni/Documents/git-projects/project577/XGBoost.pkl'

with open(clf_path, 'rb') as f:
    model.clf=pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class ClassifyReorder(Resource):
    def get(self):
        args=parser.parse_args()
        #user_query = args['query']

        # vectorize the user's query and make a prediction
        params={
            'product_id': '',
            'aisle_id' : '',
            'department_id' : '',
            'order_id' : '',
            'order_dow' : '',
            'order_hour_of_day' : '',
            'days_since_prior_order' : ''
        }
        prediction = model.predict(params)
        pred_proba = model.predict_proba(params)

        if prediction == 0:
            predict_text = "Correct"
        else:
            predict_text = "Incorrect"
        
        confidence = round(pred_proba[0], 3)

         # create JSON object
        output = {'prediction': predict_text, 'confidence': confidence}
        
        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(ClassifyReorder, '/')


if __name__ == '__main__':
    app.run(debug=True)

