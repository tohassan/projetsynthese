import sys
sys.path.insert(1, '../b_features_extraction')
from cc_preprocessing import *
from dd_models_definition import *
from gg_serving_app import myenv

import pickle

if myenv == "prod":

    ## Linear regression model -------
    with open('../../models/modelentraine_lr.pickle', 'rb') as f:
        lr_cut =  pickle.load(f)
        f.close()

    ## Polynomial linear regression model -------
    # poly features 
    with open('../../models/modelentraine_pftr.pickle', 'rb') as f:
        polyftr_cut      = pickle.load(f)
        f.close()

    # poly regressor
    with open('../../models/modelentraine_plr.pickle', 'rb') as f:
        polyreg_cut   =  pickle.load(f)
        f.close()

    ## Random forest model -----
    with open('../../models/modelentraine_rf.pickle', 'rb') as f:
        rf_cut        = pickle.load(f)
        f.close()

    ## Support vector ------
    with open('../../models/modelentraine_sv.pickle', 'rb') as f:
        supportvec_cut = pickle.load(f)
        f.close()

print("Linear regression model: -------------------------")

if myenv == "dev":
    lr_cut = linear_model.LinearRegression()
    lr_cut.fit(X_train_cut, y_train_cut)

    filename = "../../models/modelentraine_lr.pickle"
    with open(filename, 'wb') as file:
        pickle.dump(lr_cut, file)
        file.close()

linear_m(lr_cut, X_test_cut, y_test_cut)


##------

print("polynomial regression model: ---------------------")

if myenv == "prod":
    X_train_poly_cut = polyftr_cut.fit_transform(X_train_cut)
    X_test_poly_cut  = polyftr_cut.fit_transform(X_test_cut)
else:
    polyftr_cut      = PolynomialFeatures(degree=2)
    X_train_poly_cut = polyftr_cut.fit_transform(X_train_cut)
    X_test_poly_cut  = polyftr_cut.fit_transform(X_test_cut)

    filename = "../../models/modelentraine_pftr.pickle"

    with open(filename, 'wb') as file:
        pickle.dump(polyftr_cut, file)
        file.close()

    #---

    polyreg_cut      = linear_model.LinearRegression()
    polyreg_cut.fit(X_train_poly_cut, y_train_cut)

    filename = "../../models/modelentraine_plr.pickle"
    with open(filename, 'wb') as file:
        pickle.dump(polyreg_cut, file)
        file.close()

    #---

poly_m(polyreg_cut, X_test_poly_cut, y_test_cut)


##------

print("Random forest model: -----------------------------")

if myenv == "dev":
    rf_cut = RandomForestRegressor(n_estimators=50, max_features=3, max_depth=4, n_jobs=-1, random_state=1)
    rf_cut.fit(X_train_cut, y_train_cut)

    filename = "../../models/modelentraine_rf.pickle"
    with open(filename, 'wb') as file:
        pickle.dump(rf_cut, file)
        file.close()

random_f(rf_cut, X_test_cut, y_test_cut)

##------

print("Support vector: -------------------------")

if myenv == "dev":
    supportvec_cut = SVR(kernel = 'linear')
    supportvec_cut.fit(X_train_cut,y_train_cut)

    filename = "../../models/modelentraine_sv.pickle"
    with open(filename, 'wb') as file:
        pickle.dump(supportvec_cut, file)
        file.close()

support_v(supportvec_cut, X_test_cut, y_test_cut)
