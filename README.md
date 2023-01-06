IA1216-420-A51-BB Fevrier 2023
==============================

predictive maintenance - Machine Learning application 
=====================================================

Project structure
-----------------

~/predictivemaintenance
        │
        ├── data               <- The original, immutable data 
        │
        ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
        │
        ├── envdev             <- Script to run the application in developpement env 
        │
        ├── envprod            <- Script to run the application in production env 
        │
        ├── LICENSE
        │
        ├── metrics            <- Test and production metrics repository 
        │
        ├── models             <- Trained and serialized models, model predictions, or model summaries
        │
        ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
        │
        ├── README.md          <- The top-level README for developers using this project.
        │
        ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
        │                         generated with `pip freeze > requirements.txt`
        │
        ├── src                <- Source code for use in this project.
        │   │
        │   │
        │   ├── a_data_processing     <- Python code for data load and eda
        │   │
        │   ├── b_features_extraction <- Python code for preprocessing and features engineering
        │   │
        │   ├── c_models_saving       <- Python code for models definition, pickle format saving and test metrics generation 
        │   │
        │   └── d_app_serving         <- Python code of the application that serves the model, collect data and performs prediction
        │    
        └── 
