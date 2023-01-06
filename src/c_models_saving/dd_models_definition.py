import sys
sys.path.insert(1, '../b_features_extraction')
from cc_preprocessing import * 

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble      import RandomForestRegressor
from sklearn.svm           import SVR
from sklearn               import linear_model
from sklearn.linear_model  import SGDRegressor
from sklearn               import metrics
import numpy as np
from datetime import datetime
from datetime import date


def save_metrics(modeltype, ytest, predicttest):
   rmse = np.sqrt( metrics.mean_squared_error(ytest, predicttest) )
   r2   = metrics.r2_score(ytest,predicttest)

   # model,rmse,r2
   print('Type de model  : ', modeltype )
   print('RMSE test_data : ', rmse )
   print('R_2 test_data  : ', r2   )

   # Open the file in append mode. Append scores to the file
   with open("../../metrics/testmetrics/testmetrics.txt", 'a') as outfile:

        outfile.write(  datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "," + 
                                                             modeltype + "," +
                                               "{0:2.1f}".format(rmse) + "," +
                                                 "{0:2.1f}".format(r2) + "\n")


def linear_m(lm, X_test, y_test):
    predict_test = lm.predict(X_test)
    save_metrics("linear_regression", y_test, predict_test)

#polynomial model
def poly_m(polyreg, X_test_poly, y_test):
    y_test_predict = polyreg.predict(X_test_poly)
    save_metrics("polynomial_regression", y_test, y_test_predict)

#random forest model
def random_f(rf, X_test,y_test):
    y_test_predict = rf.predict(X_test)
    save_metrics("random_forest", y_test, y_test_predict)

#Support vector regressor model
def support_v(supportvec, X_test, y_test):
    y_test_predict = supportvec.predict(X_test)
    save_metrics("support_vector", y_test, y_test_predict)

