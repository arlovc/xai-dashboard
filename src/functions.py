import shap
import numpy

#shap.initjs()

"A function file contains helping functions that are called in streamlit_app.py."

def shap_force(index, 
               X_train_df, #y_train,
               explainer, 
               shap_vals):
  
    """ 

    Args:
        index (int) : sample number you want to show
        X_train_df (DataFrame): A Pandas DataFrame from the train-test-split (if it is the case) used to train the
        classifier, with column names corresponding to the feature names.
        y_train (series or array): Subset of y data used for training.
        index (int): The index of the observation of interest.
        explainer (shap explainer): A fitted shap.TreeExplainer object.
        shap_vals (array): The array of shap values.

    Returns:
        Figure: Shap force plot showing the breakdown of how the model made
            its prediction for the specified record in the training set.
    """    
    
    
    #Store model prediction and ground truth label
    #true_label = y_train.iloc[index]
    true_label = 1
    
    
    ## Assess accuracy of prediction
    if true_label == 1:
        accurate = 'Class 1'
    else:
        accurate = 'Class 0'
    
    
    ## Print output that checks model's prediction against true label
    print('***'*12)
    # Print ground truth label for row at index
    # Print model prediction for row at index
    print(f'Model Prediction: -- {accurate}')
    print('***'*12)
    print()
    print(explainer.expected_value)
    
    #print(shap_vals)
    #print(type(shap_vals))
    #print(shap_vals[true_label][index].shape)
    #print(X_train_df.iloc[index].shape, shap_vals[index][:,true_label].shape)
    
    
    ## Plot the prediction's explanation
    fig = shap.force_plot(    base_value=explainer.expected_value[1],
                              shap_values=shap_vals[true_label][index],
                              features=X_train_df.iloc[index],
                              link='logit')
    
    
    return fig