import sys
if len(sys.argv) == 1 :
   print(" ")
   print(" Error : Veuillez indiquer si vous etes en env de production [prod] ou developpement [dev] et re-executer")
   print(" ")
   exit()

myenv = sys.argv[1]

## Nous sommes alors dans l'environnement de production, 
## donc charger le model de prediction qui est sous forme picke du disque
if myenv == "prod": 
    print("======== Environnement de production detecte =======")
    import pickle 

    ## Random forest model -----
    print("======== Chargement du model en cours ... =======")
    with open('../../models/modelentraine_rf.pickle', 'rb') as f:
        rf_cut        = pickle.load(f)
        f.close()
    print("======== Chargement du model termine      =======")

elif myenv == "dev": 
    print("======== Environnement de developpement detecte =======")
    ## Nous sommes alors dans l'environnement de developpement, alors 
    ## Nous allons faire toutes les etapes (en appelant ee_testing_metrics): 
    #### - Chargement des donnees
    #### - EDA
    #### - pre-traitement
    #### - Feature engineering
    #### - Mise au point du model 
    #### - Generation et sauvegarde des metriques de test 
    #### - entregistrement du model sur le disuqe (sous format pickle) 

    import sys
    sys.path.insert(1, '../c_models_saving')
    from ee_testing_metrics import *
else:
    print(" ")
    print(" Error : Veuillez indiquer un environnement valide [prod] ou [dev] et re-executer")
    print(" ")
    exit()

import numpy as np
import pandas as pd
from flask import Flask
from flask import render_template

from flask_wtf import FlaskForm
from wtforms   import SubmitField, StringField, IntegerField,FloatField
from wtforms.validators import DataRequired

from flask import flash

from werkzeug.utils import redirect

class ProjetForm(FlaskForm):
    s2 = FloatField("Predicteur  s2:", validators=[DataRequired()])
    s3 = FloatField("Predicteur  s3:", validators=[DataRequired()])
    s4 = FloatField("Predicteur  s4:", validators=[DataRequired()])
    s7 = FloatField("Predicteur  s7:", validators=[DataRequired()])
    s8 = FloatField("Predicteur  s8:", validators=[DataRequired()])
    s9 = FloatField("Predicteur  s9:", validators=[DataRequired()])
    s11 = FloatField("Predicteur s11:", validators=[DataRequired()])
    s12 = FloatField("Predicteur s12:", validators=[DataRequired()])
    s13 = FloatField("Predicteur s13:", validators=[DataRequired()])
    s14 = FloatField("Predicteur s14:", validators=[DataRequired()])
    s15 = FloatField("Predicteur s15:", validators=[DataRequired()])
    s17 = FloatField("Predicteur s17:", validators=[DataRequired()])
    s20 = FloatField("Predicteur s20:", validators=[DataRequired()])
    s21 = FloatField("Predicteur s21:", validators=[DataRequired()])

    submit = SubmitField("    Predict    ")


app = Flask(__name__, template_folder='../../references/flask/templates')


@app.route('/')
@app.route('/index')
def index():
    projets = {'titre': 'Remaining Useful Life (RUL) Prediction'}
    return render_template('index.html', title='Accueil', mod=projets)


@app.route('/form_projet_input', methods=['GET', 'POST'])
def form_projet_input():
    projet_form = ProjetForm()

    if projet_form.validate_on_submit():
        mys2 = projet_form.s2.data
        mys3 = projet_form.s3.data
        mys4 = projet_form.s4.data
        mys7 = projet_form.s7.data
        mys8 = projet_form.s8.data
        mys9 = projet_form.s9.data
        mys11 = projet_form.s11.data
        mys12 = projet_form.s12.data
        mys13 = projet_form.s13.data
        mys14 = projet_form.s14.data
        mys15 = projet_form.s15.data
        mys17 = projet_form.s17.data
        mys20 = projet_form.s20.data
        mys21 = projet_form.s21.data

        flash('Predicteur S2:{}'.format(mys2))
        flash('Predicteur S3:{}'.format(mys3))
        flash('Predicteur S4:{}'.format(mys4))
        flash('Predicteur S7:{}'.format(mys7))
        flash('Predicteur S8:{}'.format(mys8))
        flash('Predicteur S9:{}'.format(mys9))
        flash('Predicteur S11:{}'.format(mys11))
        flash('Predicteur S12:{}'.format(mys12))
        flash('Predicteur S13:{}'.format(mys13))
        flash('Predicteur S14:{}'.format(mys14))
        flash('Predicteur S15:{}'.format(mys15))
        flash('Predicteur S17:{}'.format(mys17))
        flash('Predicteur S20:{}'.format(mys20))
        flash('Predicteur S21:{}'.format(mys21))

        flash('----------------------------------------------')

        y_pred = rf_cut.predict(pd.DataFrame(
            np.array([[mys2, mys3, mys4, mys7, mys8, mys9, mys11, mys12, mys13, mys14, mys15, mys17, mys20, mys21, ]])))
        flash('Prediction={}'.format(round(y_pred[0], 3)))

        return redirect('/index')

    #    else:
    #        flash( "{}".format("Erreur: Verifier vos parametres") )

    return render_template('form_projet_input.html',
                           title='Entrez les valeurs',
                           form=projet_form)


import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'Le renard saute la barriere'


if __name__ == '__main__':

    print("=============================================================================")
    print("=                                                                           =")
    print("=                                                                           =")
    print("=   Connectez-vous a l'URL   http://127.0.0.1:5000/                         =")
    print("=   puis entrez vos donnees pour avoir la prediction correspondante         =")
    print("=                                                                           =")
    print("=                                                                           =")
    print("=============================================================================")
    app.config.from_object(Config)
    # app.config['TESTING'] = True
    app.run(debug=True, use_reloader=False)
