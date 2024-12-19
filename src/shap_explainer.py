import pandas as pd
import shap
import numpy
from datetime import date
import os
from joblib import load as jload
from joblib import dump as jdump
import sys
import pickle 
import warnings
import xgboost
warnings.filterwarnings('ignore')

#order shap by date value
#This is just to calculate the SHAP values if they have not been calculated before

list_explainer = os.listdir(os.path.join(os.getcwd(),'explainers'))
path_model = os.path.join(os.getcwd(),'models')


#load model (just the first one)

models = os.listdir(path_model)

actual_model = [k for k in models if 'joblib' in k]
print('models: ',actual_model)
model = jload(os.path.join(path_model,actual_model[0]))

folder_implem_set = os.listdir(os.path.join(os.getcwd(), 'data/implementation_set'))
file_implem_set = [file for file in folder_implem_set if '.csv' in file][0]
path_implem_set = os.path.join(os.path.join(os.getcwd(), 'data/implementation_set'),  file_implem_set)

folder_cal_set = os.listdir(os.path.join(os.getcwd(), 'data/calibration_set'))
file_cal_set = [file for file in folder_cal_set if '.csv' in file][0]
path_cal_set = os.path.join(os.getcwd(), 'data/calibration_set',  file_cal_set)


#read explainer if found, otherwise create a new one
if(len(list_explainer) < 3):
    print('Calculating SHAP values. Might take a few minutes depending on your dataset.')
    data = pd.read_csv(path_implem_set)
    
    #background_data = pd.read_csv(path_cal_set).drop('target', axis=1)

    #get the name of the classifier and today's date for naming the shap explanations
    classifier_name = str(type(model)).lower()
    
    today = date.today()
    d2 = today.strftime("%d%m%Y")

    if("tree" in classifier_name or "forest" in classifier_name):
    
        #calculate SHAP and dump the explainer
        explainer = shap.TreeExplainer(model=model)
        shap = explainer.shap_values(data)
        
                                       #feature_perturbation="interventional",
                                       #data=background_data)
        #rf_explainer = TreeExplainer(rf_model)
        #shap_values = rf_explainer.shap_values(X)
        
        #explainer.expected_value = [0.85, 0.15]
        # check_additivity=False)
        
        
        
        
        # Textual month, day and year	
        #shap values
       # with open('explainers/shap_values'+ str(d2)+'.pickle','wb') as f:
        with open('explainers/shap_values'+ str(d2)+'.joblib','wb') as f:
            jdump(shap, f)
        
        #explainer object
        with open('explainers/shap_explainer'+ str(d2)+'.joblib','wb') as f:
            jdump(explainer, f)

    else:

        explainer = shap.KernelExplainer(model)
        shap = explainer(data)
        
        with open('explainers/shap_values'+ str(d2)+'.joblib','wb') as f:
            jdump(shap, f)
        
        #explainer object
        with open('explainers/shap_explainer'+ str(d2)+'.joblib','wb') as f:
            jdump(explainer, f)
    
    
#want to calculate also interaction values?
if str(sys.argv[0]) == 'y':
    #compute interactions
    print('You will have to wait...') 
else:
    
    print('You chose not to calculate SHAP interactions')
