from datetime import date
from pandas.io import feather_format
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.figure_factory as ff
from plotly.tools import mpl_to_plotly
#from streamlit.hashing import _CodeHasher
from PIL import Image
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
import os
from joblib import load
from pickle import load as pload
from dashboard import *
from functions import *
import warnings
import xgboost

warnings.filterwarnings("ignore")
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import copy
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder

import shap
from shap import TreeExplainer, KernelExplainer
from joblib import load as jload
from joblib import dump as jdump
import unicodedata

import scipy.stats 

st.set_page_config(layout='wide')

###small functions to be used when loading heavy dataframes in order to make use of caching decorators

@st.experimental_memo
def describe_df(df):
    
    return df.describe().iloc[1:].T

def main():

    # This is the main function to run the dashboard. The pages are named and called here to run, the dashboard object storing all the necessary information for plotting and explaining.

    # set the pages
    pages = {
        "Settings": page_settings,
        "Terms explained": page_terms_explained,
        "Data analysis": page_data_analysis,
        "Global explanations": page_global_explanations,
        "Local Explanations": page_local_explanations,
        "Confidence analysis": page_confidence_analysis,
        "Model Performance": page_performance,
        # "Models comparison":page_models_comparison,
        # "Cartography":page_map,
    }

    # add the options to the sidebar
    st.sidebar.title(":floppy_disk: Pages")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))
    st.sidebar.write("---")

    # 1: load development data, implementation set and model
    df_dev, df_implem, df_cal, df_train, model, prediction_col, fold_col, label_col = load_data()

    # 2: load dashboard class containing dashboard components if it's the first time starting the app
    if "dashboard" not in st.session_state:
        dashboard = Dashboard(
            df_dev=df_dev,
            df_implem=df_implem,
            df_cal=df_cal,
            df_train=df_train,
            model=model,
            prediction_col=prediction_col,
            fold_col=fold_col,
            label_col=label_col,
        )
        # store the dashboard in the session state with all its components for further use
        st.session_state.dashboard = dashboard

    # Display the selected page
    pages[page]()


@st.cache(allow_output_mutation=True)
def load_data():
    # This function loads all datasets and gets all the paths to the specific files."

    # get path to datasets
    files_dev = os.listdir(os.path.normpath(os.path.join("data/development_set")))
    files_implem = os.listdir(os.path.join("data/implementation_set"))
    files_cal = os.listdir(os.path.join("data/calibration_set"))
    files_train = os.listdir(os.path.join("data/train_set"))
    files_model = os.listdir(os.path.join("models"))

    # check for files that are csv or .joblib
    csv_dev = [file for file in files_dev if ".csv" in file]
    csv_implem = [file for file in files_implem if ".csv" in file]
    csv_cal = [file for file in files_cal if ".csv" in file]
    csv_train = [file for file in files_train if ".csv" in file]
    xls_train = [file for file in files_train if ".xls" in file]
    model = [file for file in files_model if ".joblib" in file or ".pickle" in file or ".pkl" in file]
    xls_cal = [file for file in files_cal if ".xls" in file]

    # check if files exist
    assert len(csv_dev) > 0, "No development set found in folder data/development_set."
    assert (len(csv_implem) > 0), "No development set found in folder data/implementation_set."
    assert len(model) > 0, "No model found in folder models/."

    # read development set
    path = os.path.join("data/development_set", csv_dev[0])
    df_dev = pd.read_csv(path)

    # read implementation set
    path = os.path.join("data/implementation_set", csv_implem[0])
    df_implem = pd.read_csv(path)

    # initialize calibration set variable in case no calibration set is provided
    df_cal = None
    # read calibration set, if provided
    if len(csv_cal) > 0:
        path = os.path.join("data/calibration_set", csv_cal[0])
        df_cal = pd.read_csv(path, header=0)

    if len(xls_cal) > 0:
        path = os.path.join("data/calibration_set", xls_cal[0])
        df_cal = pd.read_excel(path, header=0)
    
    # read training set
    if len(xls_train) > 0:
        path = os.path.join("data/train_set", xls_train[0])
        df_train = pd.read_excel(path)
    
    if len(csv_train) > 0:
        path = os.path.join("data/train_set", csv_train[0])
        df_train = pd.read_csv(path)
        
    # read model
    path_model = os.path.join("models", model[0])
    model = load(path_model)

    idx_pred = df_dev.columns.str.contains("pred")
    idx_fold = df_dev.columns.str.contains("fold")
    idx_label = ~idx_pred & ~idx_fold

    prediction_col = df_dev.columns[idx_pred][0]
    fold_col = df_dev.columns[idx_fold][0]
    label_col = df_dev.columns[idx_label][0]

    return df_dev, df_implem, df_cal, df_train, model, prediction_col, fold_col, label_col


def calculate_shap():
    # This function calculates SHAP values and stores them, together with the explainer"

    classifier_name = str(type(st.session_state.dashboard.model)).lower()

    # to be changed to dev set when it has the data? Does this make sense
    data = st.session_state.dashboard.df_implem
    today = date.today()
    d2 = today.strftime("%d%m%Y")

    if "tree" in classifier_name or "forest" in classifier_name or 'xgb' in classifier_name:

        # calculate SHAP and dump the explainer

        explainer = TreeExplainer(model=st.session_state.dashboard.model)

        shap = explainer.shap_values(data)
        
        with open("explainers/shap_values" + str(d2) + ".joblib", "wb") as f:
            jdump(shap, f)

        # explainer object
        with open("explainers/shap_explainer" + str(d2) + ".joblib", "wb") as f:
            jdump(explainer, f)

    else:

        explainer = KernelExplainer(model=st.session_state.dashboard.model)
        shap = explainer.shap_values(data)

        with open("explainers/shap_values" + str(d2) + ".joblib", "wb") as f:
            jdump(shap, f)

        # explainer object
        with open("explainers/shap_explainer" + str(d2) + ".joblib", "wb") as f:
            jdump(explainer, f)

    st.write("SHAP values calculated successfully!")


def page_confidence_analysis():
    st.title("Confidence analysis")
    st.write(
        "This page is for exploring the confidence of the model (based on the non-conformal framework) for different combinations variables. This can be useful for identifying blind spots in your data and model."
    )
    
    
    if "confidence_plots" not in st.session_state:
        st.session_state.confidence_plots = []

        
    with st.sidebar:
    # dimensionality selection:
        dimensionality_selector = st.radio("Dimensionality:", (1, 2, 3))
        prediction_selector = st.radio("Value:", ("Confidence", "Prediction"))
        prediction_types = {
            "Confidence": st.session_state.dashboard.confidence_predictions[:, 1],
            "Prediction": st.session_state.dashboard.model_predictions,
    }
        marginal_selector = st.checkbox("Show marginal distribution")
        marginal_type = None
        if marginal_selector:
            marginal_type = "histogram"

        variables = []

        for i in range(dimensionality_selector):
            # create feature selctbox for each dimensionality
            variable_selector = st.selectbox(
                "Select feature {0}".format(i),
                st.session_state.dashboard.df_implem.columns,
                0,
            )
            # determine variable type
            # var = dashboard.df_implem[variable_selector]
            # store selected variable and its type (TODO)
            variables.append(variable_selector)

        # create plot button
        generate_button = st.button("Generate plot")
        delete_button = st.button("Delete plots")

        if delete_button:
            st.session_state.confidence_plots = []

    if generate_button:
        # if dimensionality is 1, make scatterplot of that variable versus confidence/prediction
        if dimensionality_selector == 1:
            plot_conf = px.scatter(
                x=st.session_state.dashboard.df_implem[variables[0]],
                y=prediction_types[prediction_selector],
                marginal_x=marginal_type,
                labels={
                    "x": variables[0],
                    "y": "Model {0}".format(prediction_selector),
                },
                title="{0} versus {1}".format(variables[0], prediction_selector),
            )

        # if dimensionality is 2, make scatterplot of variable 1 vs variable 2 and color by confidence/prediction
        elif dimensionality_selector == 2:
            plot_conf = px.scatter(
                x=st.session_state.dashboard.df_implem[variables[0]],
                y=st.session_state.dashboard.df_implem[variables[1]],
                color=prediction_types[prediction_selector],
                marginal_x=marginal_type,
                marginal_y=marginal_type,
                # size= 1 - dashboard.confidence_predictions[:, 1]
                labels={
                    "x": variables[0],
                    "y": variables[1],
                    "color": prediction_selector,
                },
                title="{0} versus {1}".format(variables[0], variables[1]),
            )

        # if dimensionality is 3, make 3D scatterplot of 3 variables and color by confidence/prediction
        elif dimensionality_selector == 3:
            plot_conf = px.scatter_3d(
                x=st.session_state.dashboard.df_implem[variables[0]],
                y=st.session_state.dashboard.df_implem[variables[1]],
                z=st.session_state.dashboard.df_implem[variables[2]],
                color=prediction_types[prediction_selector],
                labels={
                    "x": variables[0],
                    "y": variables[1],
                    "z": variables[2],
                    "color": prediction_selector,
                },
                title="{0} versus {1} versus {2}".format(
                    variables[0], variables[1], variables[2]
                ),
            )

        st.session_state.confidence_plots.append(plot_conf)

    # plot all figures in the state list
    for plot in st.session_state.confidence_plots:
        st.write(plot)


# hack to plot force plots
def plot_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    return components.html(shap_html, height=height)


##all the local metrics


def local_metric_shap():

    # "This function generates SHAP plots, and a what-if tool that allows you to check how a change in the features values is changing the SHAP values. It is called by the local_metrics_page function and stores all plots in the session state. They can be removed as well without losing the explainer or model. "
   
    # generate and remove buttons for plots
    generate_button = st.sidebar.button("Generate plot")
    delete_button = st.sidebar.button("Remove plots")

    # delete local plots when delete button is clicked
    if delete_button:
        st.session_state.local_plots = []
        st.session_state.local_plots_whatif = []

    

    # checkbox to make the what-if analysis.
    # what_if_checkbox = st.checkbox('I wish to do a What-if analysis', value = False)
    #st.session_state.dashboard.shap_vals[1]
    
    
    #xgb and Rf have different structures, so we need to adapt the calls to the shap object
    classifier_name = str(type(st.session_state.dashboard.model)).lower()
    if 'xgb' in classifier_name:
        
        explanation_object = shap.Explanation(
            values=st.session_state.dashboard.shap_vals[
                st.session_state.selectbox_index
            ],
            base_values=st.session_state.dashboard.explainer.expected_value,
            data=st.session_state.dashboard.df_implem.iloc[
                st.session_state.selectbox_index
            ],
        )
        
    elif 'forest' in classifier_name or 'tree' in classifier_name:
        
        explanation_object = shap.Explanation(
            values=st.session_state.dashboard.shap_vals[1][
                st.session_state.selectbox_index
            ],
            base_values=st.session_state.dashboard.explainer.expected_value[1],
            data=st.session_state.dashboard.df_implem.iloc[
                st.session_state.selectbox_index
            ],
        )
    
    # adds the forceplot to the state and see below HTML wrapper to display the function
    if generate_button and ("Force Plot" in st.session_state.shap_methods):
        force_plot = shap.force_plot(explanation_object)
        st.session_state.local_plots.append(("Force Plot", force_plot))

    # creates waterfall plot
    if generate_button and ("Waterfall Plot" in st.session_state.shap_methods):
        fig, ax = plt.subplots()
        
        shap.plots.waterfall(explanation_object)

        # add to the session state for later plotting of all plots
        st.session_state.local_plots.append(("Waterfall Plot", fig))

    if generate_button and ("Decision Plot" in st.session_state.shap_methods):

        # TODO add logit switch
        # force plot, using link=logit gives you values that are not symmetrical and show controbutions in log odds. For simplicity we use probabilities directly

        display_range = slice(-1, -st.session_state.shap_max_display, -1)
        st.session_state.display_range = display_range
        fig, ax = plt.subplots()
        shap.decision_plot(
            st.session_state.dashboard.explainer.expected_value[1],
            st.session_state.dashboard.shap_vals[1][
                st.session_state.min_max_decision[0] : st.session_state.min_max_decision[1]
            ],
            feature_display_range=st.session_state.display_range,
            feature_names=st.session_state.dashboard.df_implem.columns.tolist(),
            highlight=[],
            show=False,
        )

        st.session_state.local_plots.append(("Decision Plot", fig))

    # plot local plots
    for plot in st.session_state.local_plots:
        # plot force plot with specific function to activate javascript functionality
        if plot[0] == "Force Plot":
            plot_shap(plot[1])
        else:
            st.write(plot[1])
    ######################################## what if tool

    # st.session_state.what_if_checkbox = what_if_checkbox
    # init the state which saves the what_if plots
    if "What-if Waterfall Plot" in st.session_state.shap_methods:

        st.write("---")

        # All grid configuration is done thorugh a dictionary passed as ```gridOptions``` parameter to AgGrid call.
        # You can build it yourself, or use ```gridOptionBuilder``` helper class.
        # Ag-Grid documentation can be read at https://www.ag-grid.com/documentation

        st.header("What-if tool")
        st.write(
            "Lets you edit each feature value and see the resulting change in SHAP value"
        )

        # get initial data and shape of future row transformed to matrix
        initial_columns = st.session_state.dashboard.df_implem.columns
        matrix_row_nr = round(len(initial_columns) / 10)
        extra_indices = len(initial_columns) % 10
        

        # make a dataframe out of the chosen row
        row_data = st.session_state.dashboard.df_implem.iloc[
            st.session_state.selectbox_index
        ].values
        

        #fill the lsts with Nans until a predefined length
        complete_row_data = np.append(row_data,[np.nan]*(10-extra_indices))
        complete_col_data = np.append(initial_columns,[np.nan]*(10-extra_indices))
        extra_row_data = row_data[-extra_indices:]
        extra_col_data = initial_columns[-extra_indices:]
        
        # #choose just the complete rows

        row = np.reshape(
           complete_row_data[0 : (matrix_row_nr) * 10], (matrix_row_nr,10)
        )
        columns = np.reshape(
           complete_col_data[0 : (matrix_row_nr) * 10], (matrix_row_nr,10)
        )
        
        # this will be the object holding all rows and columns in the right format
        rows_and_columns = []

        # interlacing rows with data with column names
        for x in range(0, np.shape(columns)[0]):
            
            obj_cols = columns[x].astype('str')
            obj_rows = row[x].astype('str')

            rows_and_columns.append(obj_cols)
            rows_and_columns.append(obj_rows)
        
        #add the extra final row with Nans
        
        rows_and_columns.append(extra_col_data.astype('str'))
        rows_and_columns.append(extra_row_data.astype('str'))

        #rows_and_columns
        # create a df to feed to Ag Grid
        string_columns  = ['Col'+str(x) for x in np.arange(10)]
        what_df = pd.DataFrame(data=rows_and_columns, columns=string_columns)
        
        gb = GridOptionsBuilder.from_dataframe(what_df)

        # make all columns editable
        gb.configure_default_column(editable=True)

        # build the Grid
        gridOptions = gb.build()
        #the response, main object. editing can be done here
        grid_response = AgGrid(dataframe=what_df, gridOptions=gridOptions,width='100%',height=30*len(what_df))
        
        #adding one or zero depending if there are incomplete rows or not
        if extra_indices!=0:
            add = 1
        else:
            add=0
        # Getting the even indices
        even_index = np.arange(1, (len(columns)+add) * 2 + 1, 2)
        # select the data only, no column names
        response_data = grid_response["data"].iloc[even_index]
        # flatten to a single row
        reshaped = response_data.to_numpy().flatten()
        
        # removing NaNs by selecting just de length of real columns
        reshaped = reshaped[0:len(initial_columns)]
        # now reshape in a row and remove extra bits, then transpose
        
        modif_data = pd.DataFrame(reshaped)
        modif_data = modif_data.T
        
        #calculate shap of the modified row, and cast it to float to be sure there are no other data types
        modif_shap = st.session_state.dashboard.explainer.shap_values(modif_data.astype('float64'))
        
        # show it in a waterfall plot
        fig, ax = plt.subplots()

        # using just shap values for class 1. For some reason it is a list in a list, hence the double indexing of modif_shap[1][0]
        
        classifier_name = str(type(st.session_state.dashboard.model)).lower()
            
        if ('tree' in classifier_name) or ('forest' in classifier_name):

            explanation_object_whatif = shap.Explanation(
                values=modif_shap[1][0],
                base_values=st.session_state.dashboard.explainer.expected_value[1],
                data=modif_data,
                feature_names=initial_columns,
            )

        if 'xgb' in classifier_name:
            
            #we need to take into account the different outputs between models
            modif_shap = np.array(modif_shap.tolist()[0],dtype="float64")
            modif_data = np.array(modif_data.values.tolist()[0],dtype="float64")

            explanation_object_whatif = shap.Explanation(
                values=modif_shap,
                base_values=st.session_state.dashboard.explainer.expected_value,
                data=modif_data,
                feature_names=initial_columns,
            )

        # waterfall plot
        shap.plots.waterfall(
            explanation_object_whatif,
            max_display=st.session_state.shap_max_display,
            show=False,
        )
    # if you press the button and the correct shap method is in there, it will plot it
    if generate_button and "What-if Waterfall Plot" in st.session_state.shap_methods:
        st.session_state.local_plots_whatif.append(("What-if waterfall plot", fig))
        # st.session_state.local_plots.append(("What-if waterfall plot", fig))

        # for fig in st.session_state.local_plots_whatif:
        #    st.write(fig[1])
    for plot in st.session_state.local_plots_whatif:
        st.write(plot[1])


def local_metric_proto():

    st.write("Placeholder protodash")
    st.write("---")


def local_metric_cem():

    st.write("Placeholder CEM")
    st.write("---")


def local_metric_confidence():

    # Calcutes confidence based on the non-conformal framework for one particular row, together with credibility and probability.

    # display the different kinds of predictions
    # TODO display them in a format of a table
    st.write("---")
    st.write(
        "Probability:",
        st.session_state.dashboard.get_probability_predictions()[
            st.session_state.selectbox_index
        ].round(3),
    )
    st.write("---")
    st.write("Confidence")
    conf_preds = st.session_state.dashboard.get_confidence_predictions()[
        st.session_state.selectbox_index
    ]
    st.write("Most likely class:", conf_preds[0].round(3))
    st.write("Credibility:", conf_preds[2].round(3))
    st.write("Confidence:", conf_preds[1].round(3))
    st.write("---")


##all the pages
def page_settings():
    st.title(":wrench: Settings")
    st.write("From here you can control the dashboard (data inputs, model types)")

    st.write("---")

    # TODO design the buttons here better

    # buttons
    show_states = st.button("Show session state")
    hide_states = st.button("Hide session state ")
    st.write("---")

    all_states = st.session_state

    if show_states:
        st.write(all_states)

    if hide_states:
        all_states = []
        st.write("")

    # get all files
    files_explainer = os.listdir(os.path.join("explainers"))
    if len(files_explainer) == 0:
        files_explainer[0] = "None, compute SHAP"

    # if there is a .gitkeep remove it
    if ".gitkeep" in files_explainer:
        for i, x in enumerate(files_explainer):
            if x == ".gitkeep":
                del files_explainer[i]

    # choose each type of explainer/values
    if "selectbox_shap_explainer" not in st.session_state:
        st.session_state.selectbox_shap_explainer = []

    if "selectbox_shap_vals" not in st.session_state:
        st.session_state.selectbox_shap_vals = []

    files_shap = [x for x in files_explainer if "values" in x]
    files_explainer = [x for x in files_explainer if "explainer" in x]

    st.session_state.selectbox_shap_explainer = st.selectbox(
        "Select SHAP explainer object", files_explainer
    )
    st.session_state.selectbox_shap_vals = st.selectbox(
        "Select SHAP values", files_shap
    )

    # make a button to create an explainer
    calculate_shap_button = st.button("Calculate SHAP values")
    if calculate_shap_button:
        st.write(
            "Creating new SHAP explainer, this might take a while depending on your dataset"
        )
        calculate_shap()

    # load button to load only when clicked
    load_shap_button = st.button("Load SHAP values and explainer")
    if load_shap_button:
        st.session_state.dashboard.load_shap(
            st.session_state.selectbox_shap_explainer,
            st.session_state.selectbox_shap_vals,
        )

    if isinstance(st.session_state.dashboard.df_cal, pd.DataFrame):
        calibration_target_name = st.multiselect(
            "Select or write just one calibration set target column name",
            st.session_state.dashboard.df_cal.columns,
        )
    calculate_ncf = st.button("Train non-conformal classifier")
    if calculate_ncf:
        st.session_state.dashboard.load_confidence(target_name=calibration_target_name)


def page_terms_explained():
    st.title("Terms Explained")
    st.write("On this page some of the terms used in the dashboard are explained")
    st.write("---")
    st.header("Confidence Interval")
    st.write(
        "The Confidence Interval metric (CI) is based on a non-conformal framework and shows the confidence of a certain prediction made by the model."
    )
    st.write("**Non-conformal framework**")
    st.write(
        "*Non-conformal prediction*:",
        "the set of predicted class labels that are conforming enough, given a selected significance level. For binary classification, the non-conformal prediction is always only one class, since returning a prediction set with both classes (positive and negative class are both conforming enough) or returning an empty set (both classes are too non-conforming) is not interesting. Conformance is computed in the non-conformal framework with a conformal function, which compares a data point of interest to the data points in the training set for all possible labels. If a data point is very conforming to the train set as one class but non-conforming as the other class, then confidence in the former is large. Different conformal functions are supported in the non-conformal framework.",
    )
    st.write("*Credibility*:", "The probability of conformance of the predicted label.")
    st.write(
        "*Confidence*:",
        "The inverse of the credibility of the second most conforming label. For binary classification, this equals C = 1 - the credibility of the label that is not predicted. For example: if both the positive and negative class are highly conforming, then both classes are likely and so confidence is low. If one of them has low credibility, then confidence is high.",
    )
    st.write(
        "*Example*: Given a binary classifier, the training set and a new data point, the conformance p-values for both classes is predicted as p_0 = 0.3 and p_1 = 0.2. In this case, the predicted set is {0}, since  p_0 >  p_1. The confidence of this prediction is 1 - p_1 = 0.8. The credibility of the prediction is equal to p_0: 0.3. While the confidence is reasonably high, the credibility is not very high since the negative class does not conform all that much to the train set."
    )
    st.write(
        "*Transductive and inductive conformal prediction*: for calculating the conformance a new data point is compared to a training set. This is the general approach of a transductive conformal predictor. However, in this dashboard the inductive approach is used, where instead of the training set a separate calibration set is used, which is generally smaller than the train set. This calibration set is disjoint from the training set. Using an inductive conformal predictor is less computaionally intensive, while the estimated p-values obtained this way are a good approximation of the p-values computed by the transductive approach."
    )

@st.experimental_memo
def compute_stats(df):
    # compute statistics for each column of the dataframe
    missingness = pd.Series([df[col].isna().sum() for col in df.columns])
    missingness.name = 'missingness'
    missingness.index = df.columns
    
    kurtosis = df.kurtosis()
    kurtosis.name = 'kurtosis'
    
    skew = df.skew()
    skew.name = 'skewness'
    
    variance = df.var()
    variance.name = 'variance'
    
    median = df.median()
    median.name = 'median'
    
    mad = df.mad()
    mad.name = 'MAD'
    
    stats_df = pd.concat([missingness, 
                          kurtosis,
                          skew,
                          variance,
                          median,
                          mad], axis=1)
    
    return stats_df

@st.experimental_memo
def compute_statistical_tests(df1, df2, selected_tests):
    tests = {"MWU": scipy.stats.mannwhitneyu,
             "Kol_Smir": scipy.stats.kstest}
    
    test_result = {}
    
    for test in selected_tests:
        
        test_result[test] = [tests[test](df1[col], df2[col]).pvalue if df1[col].sum() > 0 else np.nan for col in df1.columns]
        
    
    
    
    #mwus = [scipy.stats.mannwhitneyu(x=df1[col],
    #                                 y=df2[col]).pvalue if df1[col].sum() > 0 else np.nan for col in df1.columns]
    #kol = [scipy.stats.kstest(df1[col],
    #                          df2[col]).pvalue if df1[col].sum() > 0 else np.nan for col in df1.columns]
    
    
    return test_result
    

def page_data_analysis():

    st.title("Data Analysis")
    st.write(
        "On this page you can do basic datacrunching, get data counts and distributions"
    )
    st.write("---")
  
    standard_statistics = ['missingness', 
                           'kurtosis', 
                           'skewness',
                           'variance',
                           'median',
                           'MAD']
    
    statistical_tests = ['MWU', 'Kol_Smir']
    

    df_dev = st.session_state.dashboard.df_train.copy()
    df_implem = st.session_state.dashboard.df_implem.copy()
    
    describe_stats = list(df_dev.describe().T.columns)

    # all widgets:
    with st.sidebar:

        # initialize state variables
        if "chosen_statistics" not in st.session_state:
            st.session_state.plot_type = 'Statistics'
            st.session_state.chosen_statistics = standard_statistics
            st.session_state.chosen_tests = statistical_tests
            st.session_state.chosen_variables = []
            st.session_state.sort_variable = 'mean'
            st.session_state.sort_order = "Descending"
            st.session_state.df_height = 500
            st.session_state.sort_dataset = 'diff'
            st.session_state.absolute_difference = True
            st.session_state.standardize = False
            st.session_state.n_variables_slider = 5
            st.session_state.nbins_slider = 50
            st.session_state.opacity_slider = 0.75

        st.session_state.df_height = int(st.number_input("Height of DataFrame", min_value=100, max_value=3000, value=int(st.session_state.df_height)))
        st.write('---')
        # allow the user to select whether to plot automatically by top variables sorted by specific statistic,
        # or manually by selecting specific variables
        st.radio("Do you want to plot variables by the top/bottom statistics or by variable name?",
                                                options=["Statistics", "Variables"],
                                                key="statistics_or_variables",
                                                on_change=lambda: exec('st.session_state.plot_type = st.session_state.statistics_or_variables'),
                                                index=(["Statistics", "Variables"]).index(st.session_state.plot_type))
        st.write('---')
        
        # choose which standard statistics (in addition to standard Pandas describe statistics) to show
        st.multiselect(label="Standard statistics to show", 
                        options=standard_statistics, 
                        key="standard_statistics_selector",
                        on_change=lambda: exec('st.session_state.chosen_statistics = st.session_state.standard_statistics_selector'), 
                        default=st.session_state.chosen_statistics,
                        )
        
        # choose which statistical tests between train and implementation set should be shown in the dataframe
        st.multiselect("Statistical tests to show", 
                       options=statistical_tests,
                       key="statistical_tests_selector",
                       on_change=lambda: exec('st.session_state.chosen_tests = st.session_state.statistical_tests_selector'),
                       default=st.session_state.chosen_tests,
                       )
        
        if(st.session_state.plot_type == "Variables"):
            st.session_state.chosen_variables = st.multiselect("Variables to plot",
                                                                options=df_implem.columns, 
                                                                )

 
            st.write('---')
            
            
            st.session_state.sort_order = st.radio(label='Sort order', 
                                                options=['Descending', 'Ascending'],
                                                index=['Descending', 'Ascending'].index(st.session_state.sort_order))
            
            # choose to sort based on a statistic in the training set, implementation set or the difference between them
            st.session_state.sort_dataset = st.selectbox("Dataset to sort on",
                                                        options=['train', 'implem', 'diff'],
                                                        index=['train', 'implem', 'diff'].index(st.session_state.sort_dataset))
            
            # choose which statistic to sort the dataframe on. Affects the top variables to plot
            options = describe_stats + st.session_state.chosen_statistics + st.session_state.chosen_tests
            st.session_state.sort_variable = st.selectbox("Statistic to sort on",
                                            options=options,
                                            index=options.index(st.session_state.sort_variable))
            
            st.write('---')
            
            st.session_state.absolute_difference = st.checkbox('Use absolute differences',
                                                            value=st.session_state.absolute_difference)
        
            # allow standardization of variables to force them on the same scale (z-score)
            st.session_state.standardize = st.checkbox('Standardize variables',
                                                        value=st.session_state.standardize)
            
            st.write('---')
            
            st.session_state.nbins_slider = st.slider(label="Number of bins for histograms",
                                                    min_value=10, 
                                                    max_value=500, 
                                                    value=st.session_state.nbins_slider)
            st.session_state.opacity_slider = st.slider("Opacity of histograms",
                                                        min_value=0.1,
                                                        max_value=1.0,
                                                        value=st.session_state.opacity_slider)
        else:
            st.write('---')
            
            
            st.session_state.sort_order = st.radio(label='Sort order', 
                                                options=['Descending', 'Ascending'],
                                                index=['Descending', 'Ascending'].index(st.session_state.sort_order))
            
            # choose to sort based on a statistic in the training set, implementation set or the difference between them
            st.session_state.sort_dataset = st.selectbox("Dataset to sort on",
                                                        options=['train', 'implem', 'diff'],
                                                        index=['train', 'implem', 'diff'].index(st.session_state.sort_dataset))
            
            # choose which statistic to sort the dataframe on. Affects the top variables to plot
            options = describe_stats + st.session_state.chosen_statistics + st.session_state.chosen_tests
            st.session_state.sort_variable = st.selectbox("Statistic to sort on",
                                            options=options,
                                            index=options.index(st.session_state.sort_variable))
            
            st.write('---')
            
            st.session_state.absolute_difference = st.checkbox('Use absolute differences',
                                                            value=st.session_state.absolute_difference)
        
            # allow standardization of variables to force them on the same scale (z-score)
            st.session_state.standardize = st.checkbox('Standardize variables',
                                                        value=st.session_state.standardize)
            
            st.write('---')
            
            # select top features to plot
            st.session_state.n_variables_slider = st.slider(label="Number of top variable distributions to plot", 
                                                            min_value=1,
                                                            max_value=25, 
                                                            value=st.session_state.n_variables_slider)
            st.session_state.nbins_slider = st.slider(label="Number of bins for histograms",
                                                    min_value=10, 
                                                    max_value=500, 
                                                    value=st.session_state.nbins_slider)
            st.session_state.opacity_slider = st.slider("Opacity of histograms",
                                                        min_value=0.1,
                                                        max_value=1.0,
                                                        value=st.session_state.opacity_slider)
    if(st.session_state.standardize):
        # standardize variables in both dataframes
        #stats = (stats - stats.mean()) / stats.std()
        df_dev = (df_dev - df_dev.mean()) / df_dev.std()
        df_implem = (df_implem - df_implem.mean()) / df_implem.std()
    
    
    # initially show the simple stats from the Pandas describe function (# remove count statistic, since it's not very useful)
    stats_dev = describe_df(df_dev)
    stats_implem = describe_df(df_implem)
    
    # compute selected statistics
    custom_stats_dev = compute_stats(df_dev)[st.session_state.chosen_statistics]
    custom_stats_implem = compute_stats(df_implem)[st.session_state.chosen_statistics]
    
    stats_dev = pd.concat([stats_dev, custom_stats_dev], axis=1)
    stats_implem = pd.concat([stats_implem, custom_stats_implem], axis=1)
    stats_diff = stats_implem - stats_dev
    
    # show only absolute differences
    if(st.session_state.absolute_difference): 
        stats_diff = stats_diff.abs()
    
    # append custom stats to initial stats
    stats = pd.concat([stats_diff, stats_dev, stats_implem], axis=1)
    multilevel_index = pd.MultiIndex.from_product([['diff', 'train', 'implem'], stats_implem.columns])
    
    stats.columns = multilevel_index
    
    # compute statistical tests on distributions of train and test variables
    
    test_result = compute_statistical_tests(df1=df_dev, df2=df_implem, selected_tests=st.session_state.chosen_tests)
    column_order = list(stats.columns)
    test_columns = []
    for test in test_result:
        test_columns.append(('diff', test))
        stats[('diff', test)] = test_result[test] 
    column_order = test_columns + column_order 
    stats = stats[column_order]
        
    # re-order columns such that test columns show up first
    

    if(len(st.session_state.sort_variable) > 0):
        # can't sort on dev or implem if statistical test is selected to sort on
        if(st.session_state.sort_variable in statistical_tests):
             st.session_state.sort_dataset = 'diff'
        stats = stats.sort_values(by=[(st.session_state.sort_dataset, st.session_state.sort_variable)],
                                  ascending=st.session_state.sort_order=='Ascending')
      
    # round stats
    #stats = stats.round(2)

    # display stats for dev and implem set and the difference between them
    st.dataframe(stats.style.format("{:.2f}"),
                 height=int(st.session_state.df_height))
  
    if(st.session_state.plot_type == 'Statistics'):
        # based on sorting of the stats dataframe, allow selection of variables for plotting
        top_variables = stats.index[: st.session_state.n_variables_slider]
    elif(st.session_state.plot_type == 'Variables'):
        top_variables = st.session_state.chosen_variables
    
    for var in top_variables:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_dev[var], name="dev", histnorm='probability', nbinsx= st.session_state.nbins_slider))
        fig.add_trace(go.Histogram(x=df_implem[var], name="implem", histnorm='probability', nbinsx= st.session_state.nbins_slider))
        fig.update_layout(barmode='overlay', title=var)
        fig.update_traces(opacity= st.session_state.opacity_slider)
        st.write(fig)
        
    #gb = GridOptionsBuilder.from_dataframe(stats)

    # make all columns editable
    #gb.configure_default_column(editable=True)

    # build the Grid
    #gridOptions = gb.build()

    #grid_response = AgGrid(dataframe=stats, gridOptions=gridOptions)#, width="full")
    

def page_local_explanations():
    "This function contains all the local metric functions. It calls them and stores them in the state. Specific aspects related to each local metric can be edited from within each local metric function"
    # TODO decide if we use 2 columns (probably yes)

    st.title("Local Explanations")
    st.write("Local explanations based on individual samples or features.")
    st.write("---")

    # initialize local plots list
    if "local_plots" not in st.session_state:
        st.session_state.selectbox_index = 0
        st.session_state.local_plots = []
        st.session_state.local_plots_whatif = []
        st.session_state.multiselect_local_metrics = []
        st.session_state.show_row = False
        st.session_state.shap_methods = []
        st.session_state.shap_max_display = 10
        st.session_state.min_max_decision = (0,100)
        
    with st.sidebar:
        # choose the index of the row that you want to do a local explanation about
        index_options = list(range(0, len(st.session_state.dashboard.df_implem)))
        st.selectbox("Select the row index for your data", 
                     options=index_options, 
                     key="local_row_selection",
                     on_change=lambda: exec('st.session_state.selectbox_index = st.session_state.local_row_selection'),
                     index=st.session_state.selectbox_index)
     

        # display the selected row
        st.checkbox(label="Display row",
                    key="test2",
                    on_change=lambda: exec('st.session_state.show_row = st.session_state.test2')
                    )
        

        # available metrics dict pages
        metrics = {
            "SHAP": local_metric_shap,
            #'CEM': local_metric_cem,
            #'ProtoDash': local_metric_proto
        }
        if isinstance(st.session_state.dashboard.df_cal, pd.DataFrame):
            metrics["Confidence Interval"] = local_metric_confidence

        # state-saving selections
        st.multiselect(
            label="Select your metric",
            options=tuple(metrics.keys()),
            key="local_metric_selection",
            on_change=lambda: exec('st.session_state.multiselect_local_metrics = st.session_state.local_metric_selection'),
            default=st.session_state.multiselect_local_metrics,
        )
        

        # display option widgets for selected widgets
        if len(st.session_state.multiselect_local_metrics) > 0:
            # display selectors for selected metrics
            if "SHAP" in st.session_state.multiselect_local_metrics:
                SHAP_methods = [
                    "Waterfall Plot",
                    "Force Plot",
                    "Decision Plot",
                    #"What-if Waterfall Plot",
                ]

                # display selector for waterfall plot, force plot and decision plot
                st.multiselect(label="Select your SHAP method", 
                               options=SHAP_methods,
                               key="SHAP_metrics",
                               on_change=lambda: exec('st.session_state.shap_methods = st.session_state.SHAP_metrics'),
                               default=st.session_state.shap_methods
                )


                # how many features you'd like to add to each plot
                st.sidebar.slider("Number of features to show",
                                  min_value=2,
                                  max_value=st.session_state.dashboard.df_implem.columns.shape[0],
                                  value=st.session_state.shap_max_display,
                                  on_change=lambda: exec('st.session_state.shap_max_display = st.session_state.n_features_slider'),
                                  key='n_features_slider'
                )

                # get the range of rows (samples) to calculate the decision plot
                st.sidebar.slider(label="Select which rows (samples) you wish to display in the decision plot",
                                 min_value=0,
                                 max_value=len(st.session_state.dashboard.df_implem),
                                 key="decision_plot_slider",
                                 on_change=lambda: exec('st.session_state.min_max_decision = st.session_state.decision_plot_slider'),
                                 value=st.session_state.min_max_decision,
                )
        
    if st.session_state.show_row:
        st.dataframe(st.session_state.dashboard.df_implem.iloc[st.session_state.selectbox_index])
        
    # run all selected metric functions
    if len(st.session_state.multiselect_local_metrics) > 0:
        # create figures of selected metrics after button click
        for metric in st.session_state.multiselect_local_metrics:
            metrics[metric]()


def page_global_explanations():

    # presents global explanations (all samples) Options are PDPs, Confidence, SHAP summary. Confidence can be shown in different plot types.

    st.title(":earth_asia: Global Explanations")
    st.write("Global explanations: rule-based, surrogate models, etc")
    st.write("---")

    # initialize state variables
    if "global_plots" not in st.session_state:
        st.session_state.global_plots = []
        st.session_state.max_display_summary = 10
        st.session_state.metric_idx = 0
        st.session_state.grid_resolution = 10
        st.session_state.n_top_features = 10
   
    # bug in beeswarm plot code changes shap values; provide copy of shap values to prevent shap values from changing!
    # Create explanation object that contains all information needed for SHAP matplotlib plots

    #xgb and Rf give different data structures, take that into consideration
    classifier_name = str(type(st.session_state.dashboard.model)).lower()
    
    if ('tree' in classifier_name) or ('forest' in classifier_name):

        explanation_object = shap.Explanation(
            values= st.session_state.dashboard.shap_vals[1],
            base_values=st.session_state.dashboard.explainer.expected_value[1],
            data=st.session_state.dashboard.df_implem.values,
            feature_names=st.session_state.dashboard.df_implem.columns,
        )

    if 'xgb' in classifier_name:
        explanation_object = shap.Explanation(
            values= st.session_state.dashboard.shap_vals,
            base_values=st.session_state.dashboard.explainer.expected_value,
            data=st.session_state.dashboard.df_implem.values,
            feature_names=st.session_state.dashboard.df_implem.columns,
        )
        

    # define variables that can be plotted in histograms and scatterplots
    variable_options = ["Predictions"]
    # possible variables for plotting
    variables = {
        "Predictions": st.session_state.dashboard.get_probability_predictions()
    }
    # add confidence and credibility if calibration set was provided
    if isinstance(st.session_state.dashboard.df_cal, pd.DataFrame):
        # variable options
        variable_options.append("Confidence")
        variable_options.append("Credibility")
        # variable values
        variables[
            "Confidence"
        ] = st.session_state.dashboard.get_confidence_predictions()[:, 1]
        variables[
            "Credibility"
        ] = st.session_state.dashboard.get_confidence_predictions()[:, 2]

    # available metrics as available variables plus SHAP and partial dependence plots
    st.session_state.metrics = variable_options + ["SHAP beeswarm summary", "SHAP Partial Dependence", "SHAP mean absolute", "Partial Dependence"]

    # menu for selecting plot type
    plot_type_options = [
        "histogram",
        "scatterplot"
    ]     
        
    # standard input for plot
    plot_name = st.sidebar.text_input(label="Give your plot a name", value="")
    st.sidebar.selectbox(
        label="Select your metric", 
        options=st.session_state.metrics, 
        key="global_metric_selection",
        on_change=lambda: exec('st.session_state.metric_idx = st.session_state.metrics.index(st.session_state.global_metric_selection)'),
        index=st.session_state.metric_idx
    )

    metric = st.session_state.metrics[st.session_state.metric_idx]
    # st.session_state.metric_idx = metrics.index(metric)
    # allow plot type selection, depending on selected metric. This IF statement controls what happens in the sidebar for the given plot type. 
    # If there are not sidebar controls, then ONLY the next if-statement controls the plotting
    with st.sidebar:

        if metric == "Predictions" or metric == "Confidence" or metric == "Credibility":
            # menu for selecting plot type
            plot_type_options = ["histogram", "scatterplot"]
            # plot type selectbox
            plot_type = st.selectbox(label="Select plot type", 
                                     options=plot_type_options,
                                     )

            # set x-axis to selected variable metric
            plot_xaxis = metric

            # only display bin slider for histograms
            if plot_type == "histogram":
                # select variable on the x-axis
                nbins = st.slider(label="nbins", min_value=2, max_value=250, value=50)

            elif plot_type == "scatterplot":
                plot_yaxis = st.selectbox(
                    label="Select Y-axis variable", options=variable_options
                )           

        elif metric == "SHAP beeswarm summary" or metric == "SHAP mean absolute":
            st.session_state.max_display_summary = st.slider(
                "Number of top features to show",
                key="n_top_features",
                on_change=lambda: exec('st.session_state.n_top_features = st.session_state.n_top_features'),
                value=st.session_state.n_top_features,
                max_value=st.session_state.dashboard.df_implem.shape[1],

            )
        elif metric == "SHAP Partial Dependence":
                
                pdp_shap_variable1 = st.selectbox(
                label="Select variable",
                options=st.session_state.dashboard.df_implem.columns,
            )
                pdp_shap_variable2 = st.selectbox(
                label="Select interaction variable",
                options=st.session_state.dashboard.df_implem.columns,
            )

        elif metric == "Partial Dependence":
            
            pdp_dimensionality_selector = st.radio("Dimensionality:", (1, 2))
            pdp_variables = []

            for i in range(pdp_dimensionality_selector):
                # create feature selctbox for each dimensionality
                
                variable_selector = st.selectbox(
                    "Select feature {0}".format(i),
                    st.session_state.dashboard.df_implem.columns.tolist(),
                    0,
                )
                # store selected variable and its type (TODO)
                pdp_variables.append(variable_selector)

            marginal_selector = st.checkbox("Show marginal distribution")
            marginal_type = None

            if marginal_selector:
                marginal_type = "histogram"

            show_individual = st.checkbox("Show individual plots")

            td_plot = st.sidebar.checkbox('Do you want to plot in 3D? You need dimensionality 2')

            if show_individual:
                n_rows = st.slider(
                    label="number of individual plots",
                    min_value=1,
                    max_value=st.session_state.dashboard.df_implem.shape[0],
                    value=50,
                    )
                opacity = st.slider(label="Opacity of individual plots", 
                                                min_value=0.1,
                                                max_value=1.0,
                                                value=0.1)
            else:
                opacity = 0.1


        st.sidebar.slider(
            label="number of grid points", 
            min_value=2, 
            max_value=20, 
            key="grid_resolution",
            on_change=lambda: exec('st.session_state.grid_resolution = st.session_state.grid_resolution'),
            value=st.session_state.grid_resolution
        )

        generate_button = st.sidebar.button("Generate plot")
        delete_button = st.sidebar.button("Remove plots")

    # if a generate button is clicked, generate a new plot
    if generate_button:
        # generate histogram or scatterplot if one of the variables is selected
        if metric in variable_options:
            if plot_type == "histogram":
                # generate default name
                if plot_name == "":
                    plot_name = plot_type + " of " + plot_xaxis
                new_plot = px.histogram(
                    variables[plot_xaxis], nbins=nbins, title=plot_name
                )

            elif plot_type == "scatterplot":
                # generate default name
                if plot_name == "":
                    plot_name = (
                        plot_type + " of " + plot_xaxis + " versus " + plot_yaxis
                    )
                new_plot = px.scatter(
                    x=variables[plot_xaxis],
                    y=variables[plot_yaxis],
                    labels={"x": plot_xaxis, "y": plot_yaxis},
                    title=plot_name,
                )
        elif metric =="SHAP mean absolute":
            
            new_plot, ax = plt.subplots()
            shap.summary_plot(explanation_object.values,
                              max_display=st.session_state.max_display_summary,
                              plot_type='bar',
                              feature_names=st.session_state.dashboard.df_implem.columns)

        elif metric == "SHAP beeswarm summary":
            # pre-compute SHAP summary for now
            new_plot, ax = plt.subplots()
            shap.summary_plot(
                explanation_object, max_display=st.session_state.max_display_summary
            )
            
        elif metric == "SHAP Partial Dependence":
            # plot shap dependency plots for the first variable and show interaction effects with another variable
            new_plot, ax= plt.subplots()
            shap.dependence_plot(pdp_shap_variable1,
                                 explanation_object.values,
                                 st.session_state.dashboard.df_implem,
                                 interaction_index=pdp_shap_variable2,
                                 ax=ax)

        elif metric == "Partial Dependence":

            if pdp_dimensionality_selector == 1:
                
                pdp_result = partial_dependence(
                    estimator=st.session_state.dashboard.model,
                    X=st.session_state.dashboard.df_implem,
                    features=[
                        st.session_state.dashboard.df_implem.columns.tolist().index(
                            pdp_variables[0])
                        
                    ],
                    kind="both",
                    grid_resolution=st.session_state.grid_resolution,

                )

                average = pdp_result["average"].flatten()
                individual = pdp_result["individual"][0]
                x_values = pdp_result["values"][0].flatten()

                # initialize figure
                new_plot = go.Figure(
                    layout=go.Layout(
                        title=go.layout.Title(
                            text="Partial Dependence Plot of " + pdp_variables[0]
                        )
                    )
                )

                # add average plot
                new_plot.add_trace(
                    go.Line(
                        x=x_values, y=average, line=dict(color="black"), name="average"
                    ),
                )

                if show_individual:
                    # sample random individual PDP plots
                    indices = np.arange(individual.shape[0])
                    indices = np.random.choice(a=indices,
                                            size=n_rows,
                                            replace=False)
                    
                
                    # add individual plots
                    for i in indices:
                        new_plot.add_trace(
                            go.Line(
                                x=x_values,
                                y=individual[i],
                                line=dict(color="gray"),
                                opacity=opacity,
                                showlegend=False,
                            )
                        )
                new_plot.update_layout(
                    xaxis_title=pdp_variables[0],
                    yaxis_title="Model output",
                    yaxis_range=[0, 1],
                )

            else:
                if td_plot ==True:
                    #TODO plot 3d wiht plotly

                    pdp = partial_dependence(
                        st.session_state.dashboard.model, 
                        st.session_state.dashboard.df_implem,
                        features=pdp_variables,
                        kind="average", 
                        grid_resolution=st.session_state.grid_resolution

                    )
                    new_plot = plt.figure()

                    XX, YY = np.meshgrid(pdp["values"][0], pdp["values"][1])
                    Z = pdp.average[0].T
                    ax = Axes3D(new_plot)
                    new_plot.add_axes(ax)
                    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor="k")
                    ax.set_xlabel(pdp_variables[0])
                    ax.set_ylabel(pdp_variables[1])
                    # pretty init view
                    ax.view_init(elev=22, azim=122)
                    plt.colorbar(surf)
                    plt.subplots_adjust(top=0.9)
                else:
                    
                    #plot matplotlib 2d
                    #TODO plot with plotly
                    features = pdp_variables
                    PartialDependenceDisplay.from_estimator(st.session_state.dashboard.model, st.session_state.dashboard.df_implem,[tuple(features)], grid_resolution = st.session_state.grid_resolution)

                    new_plot = plt.gcf()

    # if generate button was clicked, simply add plot to plotlist
    if generate_button:
        # don't persist matplotlib plots or streamlit times out
        # if(plot_type != "SHAP summary"):
        st.session_state.global_plots.append(new_plot)

    # reset plotting state variables
    if delete_button:
        st.session_state.global_plots = []
        
    col1, col2 = st.columns(2)

    # plot both static plots and side-by-side plots
    for i, plot in enumerate(st.session_state.global_plots):
        # plot even-numbered plots on the left, odd-numbered plots on the right
        if(i % 2 == 0):
            col1.write(plot)
        else:
            col2.write(plot)
       

def page_map():

    st.title(":earth_americas: Cartography")
    st.write("View your data on a map, visualise spatial dependencies")
    st.write("---")


def page_performance():

    st.title(":chart_with_upwards_trend: Performance")
    st.write("All your performance metrics: AUC ROC, PR curve, confusion matrix a.o")
    st.write("---")
    
    with st.sidebar:
        auc_option = st.radio(label="AUC display", options=["All", "Mean"])
        # add a slider for setting the threshold
        threshold = st.slider(label="threshold", min_value=0.0, max_value=1.0, value=0.5)   
    # create and plot ROC curve
    st.write(st.session_state.dashboard.create_ROC_curve(option=auc_option))
    # create and plot PR curve
    st.write(st.session_state.dashboard.create_PR_curve(option=auc_option))
    
    # create and plot confusion matrix
    st.write(st.session_state.dashboard.create_confusion_matrix(threshold=threshold))
    # create and display classification metrics
    metrics = st.session_state.dashboard.compute_metrics(threshold=threshold)
    st.write("Accuracy:", metrics["mean_accuracy"])
    st.write("Precision:", metrics["mean_precision"])
    st.write("Recall:", metrics["mean_recall"])
    st.write("F1:", metrics["mean_f1"])
    # create and plot precision-at-K
    st.write(st.session_state.dashboard.compute_top_k_plots())


def page_models_comparison():

    st.title("Model comparison")
    st.write("Compare two model explanations or performances. Beta")
    st.write("---")


if __name__ == "__main__":
    main()
