from model import XGBoostModel
import pandas as pd
from sklearn.model_selection import train_test_split


def build_model():
    model = XGBoostModel()

    with open('/Users/shwetasaloni/Desktop/ISDS 577/Final_dataset.csv') as f:
        data=pd.read_csv(f, sep='\t')
    X = data[['product_id', 'aisle_id', 'department_id', 'order_id', 'order_dow', 'order_hour_of_day','days_since_prior_order']]
    y = data['reordered']
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.train(X_train, y_train)
    print('Model training complete')

    model.pickle_clf()
    
    model.feature_impo(X_train)
    model.plot_importanceFeatures()
    model.plot_roc(X_test, y_test, X_test.shape[0], y_test.shape[0])





