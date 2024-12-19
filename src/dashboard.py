import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import numpy as np
import pandas as pd

import getpass
import pickle
from joblib import load, dump
import os

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

import nonconformist
from nonconformist.icp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc
from nonconformist.nc import NcFactory
import warnings
import streamlit as st

warnings.filterwarnings("ignore")


class Dashboard:
    def __init__(
        self,
        df_dev,
        df_implem,
        df_cal,
        df_train,
        model,
        prediction_col,
        fold_col,
        label_col,
    ):

        """
        A wrapper class that contains the model, different datasets, metrics and functions to compute
        these metrics.

        Parameters:
        df_dev -- Pandas DataFrame development set containing model predictions, ground-truth labels
                  and fold indicator for each row in the training set
        df_implem -- Pandas DataFrame implementation set containing the data matrix of unseen data points
        df_cal -- Pandas DataFrame calibration set containing a subset of the training dataset with both
                  features and targets
        model -- fitted Scikit-learn Estimator
        prediction_col -- str specifying which column in df_dev indicates the model predictions
        fold_col -- str specifying which column in df_dev indicates the folds
        label_col --  str specifying which column in df_dev indicates the ground-truth labels
        """

        self.df_dev = df_dev
        self.df_implem = df_implem
        
        self.model = model
        self.model.enable_categorical  = True
        self.model.predictor = 'auto'
        self.df_cal = df_cal
        self.df_train = df_train
        
        # make sure training set and implementation set columns have the same column names
        self.df_train.columns = self.df_implem.columns

        self.prediction_col = prediction_col
        self.fold_col = fold_col
        self.label_col = label_col
        # compute number of folds (K)
        self.K = self.df_dev[fold_col].unique().size
        self.threshold = 0.5
        self.comp_top_k = self.compute_top_k_plots()
        
        self.model_predictions = self.model.predict_proba(self.df_implem)[:, 1]
        self.sorting_idx = np.flip(np.argsort(self.model_predictions))

        #sort model predictions and implementation set according to model predictions in
        #descending order
        self.model_predictions = self.model_predictions[self.sorting_idx]
        self.df_implem = self.df_implem.iloc[self.sorting_idx]
        self.df_implem = self.df_implem.set_index(self.model_predictions)

    def load_shap(self, shap_explainer, shap_values):
        """
        Load a trained SHAP explainer and corresponding shap values

        Parameters:
        shap_explainer -- path to the pickled shap explainer object
        shap_values -- path to the pickled shap values array
        """

        shap_explainer_path = os.path.join(os.getcwd(), "explainers", shap_explainer)
        shap_values_path = os.path.join(os.getcwd(), "explainers", shap_values)

        # unpickle shap values and explainer
        with open(shap_explainer_path, "rb") as f:
            self.explainer = load(shap_explainer_path)

        # explainer object
        with open(shap_values_path, "rb") as f:
            self.shap_vals = load(shap_values_path)

        # sort shap values according to model predictions
        #self.shap_vals[1] = self.shap_vals[1][self.sorting_idx]

        st.write("SHAP values and explanation object successfully loaded!")

    def load_confidence(self, target_name):
        """
        Create and fit a non-conformal classifier from the nonconformist library
        to the provided calibration set.

        Parameters:
        target_name -- str specifying which column name in the calibration set refers
                       to the target column
        
        """

        # create non-conformal scorer if calibration set is present
        if isinstance(self.df_cal, pd.DataFrame):
            
            # create inductive non-conformal classifier
            nc_model = NcFactory.create_nc(self.model)
            self.icp = IcpClassifier(nc_model)

            
            #calibrate nc classifier on calibration set. #TODO If this gives an ERROR, then switch target_name[0] to target_name
            self.icp.calibrate(
                x=self.df_cal.drop(target_name[0], axis=1),
                y=self.df_cal[target_name[0]],
            )

            # compute confidence predictions
            self.confidence_predictions = self.icp.predict_conf(self.df_implem.values)
            #self.confidence_predictions = self.confidence_predictions[self.sorting_idx]
            st.write("Non-conformal classifier trained!")

    def get_probability_predictions(self):
        # return self.model.predict_proba(self.df_implem)[:, 1]
        return self.model_predictions

    def get_confidence_predictions(self):
        # return self.icp.predict_conf(self.df_implem.values)
        return self.confidence_predictions

    def create_prediction_dist(self, nbins):
        """
        Creates a Plotly Express histogram of model predictions on
        the implementation set.

        Returns the created histogram Plotly figure.

        Parameters:
        nbins -- int indicating the number of bins to use in the
                 histogram
        """

        # create a plot showing the implementation set prediction distribution
        predictions = self.get_probability_predictions()
        comp_dist_preds = px.histogram(
            predictions,
            range_x=[0.0, 1.0],
            nbins=nbins,
            title="Distribution of model predictions on implementation set",
        )
        return comp_dist_preds

    def create_confidence_dist(self, nbins):
        """
        Creates a Plotly Express histogram of model confidence on
        the implementation set.

        Returns the created histogram Plotly figure.

        Parameters:
        nbins -- int indicating the number of bins to use in the
                 histogram
        """

        # create a plot showing the implementation set prediction distribution
        predictions = self.get_confidence_predictions()
        comp_dist_preds = px.histogram(
            predictions[:, 1],
            range_x=[0.0, 1.0],
            nbins=nbins,
            title="Distribution of model confidence on implementation set",
        )
        return comp_dist_preds

    def create_ROC_curve(self, option):
        if option == "All":
            return self.create_ROC_curve_all()
        elif option == "Mean":
            return self.create_ROC_curve_mean()

    def create_PR_curve(self, option):
        if option == "All":
            return self.create_PR_curve_all()
        elif option == "Mean":
            return self.create_PR_curve_mean()

    def create_ROC_curve_all(self):
        """
        Creates ROC curves for all
        folds found in the development set.

        Returns a Plotly Go figure containing all the ROC
        curves.
        """

        # Create an empty figure, and iteratively add new lines for each fold
        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

        for i in range(self.K):
            # get all rows for fold i
            df_i = self.df_dev[self.df_dev[self.fold_col] == i]
            y_true = df_i[self.label_col]
            y_score = df_i[self.prediction_col]
            # compute the AUC curve
            fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
            auc_score = roc_auc_score(y_true=y_true, y_score=y_score)
            name = "Fold {0} (AUC={1})".format(i + 1, round(auc_score, 3))
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            width=700,
            height=500,
        )

        return fig

    def create_ROC_curve_mean(self):
        """
        Creates ROC curves for all
        folds found in the development set.

        Returns a Plotly Go figure containing the mean
        of the ROC curves, together with curves that are
        1 and 2 standard deviations from the mean
        """

        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

        tprs = []
        auc_scores = []

        # take the first pr curve's points as the number of x points
        df_1 = self.df_dev[self.df_dev[self.fold_col] == 1]
        y_true = df_1[self.label_col]
        y_score = df_1[self.prediction_col]
        # compute the AUC curve
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)

        n_points = fpr.shape[0]
        step = 1 / n_points
        # initialize the fprs
        x = np.arange(0, 1 + step, step)

        for i in range(self.K):
            # get all rows for fold i
            df_i = self.df_dev[self.df_dev[self.fold_col] == i]
            y_true = df_i[self.label_col]
            y_score = df_i[self.prediction_col]
            # compute the AUC curve
            fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
            auc_score = roc_auc_score(y_true=y_true, y_score=y_score)
            # name = "Fold {0} (AUC={1})".format(i+1, auc_score)
            # interpolate the curve for the specified fprs
            tpr_interp = np.interp(x=x, xp=fpr, fp=tpr)
            tpr_interp[tpr_interp > 1] = 1
            tpr_interp[tpr_interp < 0] = 0

            tprs.append(tpr_interp)

            auc_scores.append(auc_score)
            # fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        # average the scores
        auc_score_mean = np.mean(auc_scores)
        auc_score_sd = np.std(auc_scores)
        # average the curves
        tprs_mean = np.mean(tprs, axis=0)
        tprs_sd = np.std(tprs, axis=0)

        name = "mean AUC={0}".format(round(auc_score_mean, 3))
        # add mean curve
        fig.add_trace(go.Scatter(x=x, y=tprs_mean, name=name, mode="lines"))

        tprs_upper_sd1 = tprs_mean + tprs_sd
        tprs_under_sd1 = tprs_mean - tprs_sd

        tprs_upper_sd2 = tprs_mean + (2 * tprs_sd)
        tprs_under_sd2 = tprs_mean - (2 * tprs_sd)

        # clamp values between 0 and 1
        tprs_upper_sd1[tprs_upper_sd1 > 1] = 1
        tprs_upper_sd1[tprs_upper_sd1 < 0] = 0

        tprs_under_sd1[tprs_under_sd1 > 1] = 1
        tprs_under_sd1[tprs_under_sd1 < 0] = 0

        tprs_upper_sd2[tprs_upper_sd2 > 1] = 1
        tprs_upper_sd2[tprs_upper_sd2 < 0] = 0

        tprs_under_sd2[tprs_under_sd2 > 1] = 1
        tprs_under_sd2[tprs_under_sd2 < 0] = 0

        # fill area between upper and lower area, defined by SD
        fig.add_trace(
            go.Scatter(
                x=x,
                y=tprs_upper_sd1,
                fill=None,
                mode="lines",
                line_color="blue",
                name="+ 1 SD",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=tprs_under_sd1,
                fill="tonexty",  # fill area between trace0 and trace1
                mode="lines",
                line_color="blue",
                name="- 1 SD",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=tprs_upper_sd2,
                fill=None,  # fill area between trace0 and trace1
                mode="lines",
                line_color="indigo",
                name="+ 2 SD",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=tprs_under_sd2,
                fill="tonexty",  # fill area between trace0 and trace1
                mode="lines",
                line_color="indigo",
                name="- 2 SD",
            )
        )

        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            width=700,
            height=500,
        )

        return fig

    def create_PR_curve_all(self):
        """
        Creates precision-recall curves for all
        folds found in the development set.

        Returns a Plotly Go figure containing all the
        precision-recall curves.
        """
        # Create an empty figure, and iteratively add new lines for each fold
        fig = go.Figure()
        prop_pos = self.df_dev[self.label_col].mean()
        fig.add_shape(
            type="line", line=dict(dash="dash"), x0=0, x1=1, y0=prop_pos, y1=prop_pos
        )

        for i in range(self.K):
            # get all rows for fold i
            df_i = self.df_dev[self.df_dev[self.fold_col] == i]
            y_true = df_i[self.label_col]
            y_score = df_i[self.prediction_col]
            # compute the AUC curve
            precision, recall, _ = precision_recall_curve(
                y_true=y_true, probas_pred=y_score
            )
            auc_score = auc(x=recall, y=precision)
            name = "Fold {0} (AUC={1})".format(i + 1, round(auc_score, 3))
            fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode="lines"))

        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])

        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            width=700,
            height=500,
        )

        return fig

    def create_PR_curve_mean(self):
        """
        Creates precision-recall curves for all
        folds found in the development set.

        Returns a Plotly Go figure containing the mean
        of the precision-recall curves, together with
        curves that are 1 and 2 standard deviations from
        the mean.
        """

        fig = go.Figure()
        prop_pos = self.df_dev[self.label_col].mean()
        fig.add_shape(
            type="line", line=dict(dash="dash"), x0=0, x1=1, y0=prop_pos, y1=prop_pos
        )

        precisions = []
        auc_scores = []

        # take the first pr curve's points as the number of x points
        df_1 = self.df_dev[self.df_dev[self.fold_col] == 1]
        y_true = df_1[self.label_col]
        y_score = df_1[self.prediction_col]
        # compute the pr curve
        precision, recall, _ = precision_recall_curve(
            y_true=y_true, probas_pred=y_score
        )

        n_points = recall.shape[0]
        step = 1 / n_points
        # initialize the fprs
        x = np.arange(0, 1 + step, step)

        for i in range(self.K):
            # get all rows for fold i
            df_i = self.df_dev[self.df_dev[self.fold_col] == i]
            y_true = df_i[self.label_col]
            y_score = df_i[self.prediction_col]
            # compute the AUC curve
            precision, recall, _ = precision_recall_curve(
                y_true=y_true, probas_pred=y_score
            )
            precision = np.flip(precision)
            recall = np.flip(recall)
            auc_score = auc(x=recall, y=precision)
            # name = "Fold {0} (AUC={1})".format(i+1, auc_score)
            # interpolate the curve for the specified fprs
            # precision_interp = np.interp(x=x, xp=precision, fp=recall)
            precision_interp = np.interp(x=x, xp=recall, fp=precision)
            # clamp values between 0 and 1
            precision_interp[precision_interp > 1] = 1
            precision_interp[precision_interp < 0] = 0

            precisions.append(precision_interp)

            auc_scores.append(auc_score)
            # fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        # average the scores
        auc_score_mean = np.mean(auc_scores)
        auc_score_sd = np.std(auc_scores)
        # average the curves
        precisions_mean = np.mean(precisions, axis=0)
        precisions_sd = np.std(precisions, axis=0)

        name = "mean AUC={0}".format(round(auc_score_mean, 3))

        # add mean curve
        fig.add_trace(go.Scatter(x=x, y=precisions_mean, name=name, mode="lines"))

        precisions_upper_sd1 = precisions_mean + precisions_sd
        precisions_under_sd1 = precisions_mean - precisions_sd

        precisions_upper_sd2 = precisions_mean + (2 * precisions_sd)
        precisions_under_sd2 = precisions_mean - (2 * precisions_sd)

        # clamp values between 0 and 1
        precisions_upper_sd1[precisions_upper_sd1 > 1] = 1
        precisions_upper_sd1[precisions_upper_sd1 < 0] = 0

        precisions_under_sd1[precisions_under_sd1 > 1] = 1
        precisions_under_sd1[precisions_under_sd1 < 0] = 0

        precisions_upper_sd2[precisions_upper_sd2 > 1] = 1
        precisions_upper_sd2[precisions_upper_sd2 < 0] = 0

        precisions_under_sd2[precisions_under_sd2 > 1] = 1
        precisions_under_sd2[precisions_under_sd2 < 0] = 0

        # fill area between upper and lower area, defined by SD
        fig.add_trace(
            go.Scatter(
                x=x,
                y=precisions_upper_sd1,
                fill=None,
                mode="lines",
                line_color="blue",
                name="+ 1 SD",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=precisions_under_sd1,
                fill="tonexty",  # fill area between trace0 and trace1
                mode="lines",
                line_color="blue",
                name="- 1 SD",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=precisions_upper_sd2,
                fill=None,  # fill area between trace0 and trace1
                mode="lines",
                line_color="indigo",
                name="+ 2 SD",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=precisions_under_sd2,
                fill="tonexty",  # fill area between trace0 and trace1
                mode="lines",
                line_color="indigo",
                name="- 2 SD",
            )
        )

        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])

        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            width=700,
            height=500,
        )

        return fig

    def create_confusion_matrix(self, threshold):
        """
        Create a summed confusion matrix plot. A confusion
        matrix is created for each fold and subsequently
        added to the total.

        returns the Plotly figure.

        Parameters:
        threshold -- float specifying the cutoff point for
                     classification. Predictions at or above
                     the threshold are set to 1, predictions
                     below the threshold are set to 0.
        """

        # compute and sum confusion matrix for each fold
        conf_matrix_summed = np.zeros((2, 2))
        for i in range(self.K):
            # get all rows for fold i
            df_i = self.df_dev[self.df_dev[self.fold_col] == i]
            y_true = df_i[self.label_col]
            y_score = df_i[self.prediction_col]
            # apply threshold to predictions
            y_score = y_score.apply(lambda x: 1 if x >= threshold else 0)
            conf_matrix = confusion_matrix(y_true, y_score)

            conf_matrix_summed += conf_matrix

        # create confusion matrix plot
        x_label = ["Predicted 0", "Predicted 1"]
        y_label = ["Actual 1", "Actual 0"]

        # swap confusion matrix rows to make the figure correct
        conf_matrix_summed[[0, 1]] = conf_matrix_summed[[1, 0]]

        fig = ff.create_annotated_heatmap(
            conf_matrix_summed, x=x_label, y=y_label, colorscale="Reds"
        )
        fig["data"][0]["showscale"] = True

        fig.update_layout(title_text="Confusion matrix", width=700, height=500)

        return fig

    def compute_precision_at_k(self, df, start=10, step=5):
        """
        Compute the recall at the top of the predictions to measure
        how well the model can catch events in the highest predictions.

        Parameters:
        df -- Pandas DataFrame containing a set of predictions of a
              specific fold
        start -- int specifying the first top to compute
        step -- int specifying the how many tops to skip
        limit -- int specifying the final top to compute
        """

        precisions = []
        recall = 0

        while recall < 1:
            # take the top-X
            top_k = df.iloc[:start]
            precision = precision_score(
                y_true=top_k[self.label_col], y_pred=np.ones(top_k.shape[0])
            )
            precisions.append(precision)
            
            recall = top_k[self.label_col].sum() / df[self.label_col].sum()

            start += step

        return precisions

    def compute_recall_at_k(self, df, start=10, step=5):
        """
        Compute the recall at the top of the predictions to measure
        how well the model can find all the events in the highest
        predictions.

        Parameters:
        df -- Pandas DataFrame containing a set of predictions of a
              specific fold
        start -- int specifying the first top to compute
        step -- int specifying the how many tops to skip
        limit -- int specifying the final top to compute
        """
        recalls = []
        recall = 0

        while recall < 1:
            # take the top-X
            top_k = df.iloc[:start]
            # divide the amount of targets found by the total amount of targets
            recall = top_k[self.label_col].sum() / df[self.label_col].sum()
            recalls.append(recall)

            start += 5

        return recalls

    def compute_top_k_plots(self):
        """
        Compute precision@k and recall@k plots. These are calculated
        separately for each fold, then averaged.

        Returns the plotly barchart containing both averaged plots.
        """

        mean_precision = []
        mean_recall = []

        start = 10
        step = 5
        #limit = 100

        for i in range(self.K):
            # get all rows for fold i
            df_i = self.df_dev[self.df_dev[self.fold_col] == i]
            # sort test set in descending order based on prediction
            df_i = df_i.sort_values(by=self.prediction_col, ascending=False)
            # compute precision at k
            precision_at_k = self.compute_precision_at_k(
                df=df_i, start=start, step=step
            )
            # compute recall at k
            recall_at_k = self.compute_recall_at_k(
                df=df_i, start=start, step=step
            )

            mean_precision.append(precision_at_k)
            mean_recall.append(recall_at_k)
        
        # find the smallest prediction top
        min_top = min([len(precisions) for precisions in mean_precision])
        # make every top the same size so they can be averaged -  0:min_top
        mean_precision = [precisions[:min_top] for precisions in mean_precision]
        mean_recall = [recalls[:min_top] for recalls in mean_recall]

        mean_precision = np.mean(mean_precision, axis=0)
        mean_recall = np.mean(mean_recall, axis=0)

        mean_precision = pd.Series(mean_precision)
        mean_recall = pd.Series(mean_recall)

        Xs = pd.Series(np.arange(start=start, step=step, stop=step * min_top))

        df = pd.DataFrame({"Precision": mean_precision, "Recall": mean_recall, "X": Xs})

        fig = px.bar(df, x="X", y=["Precision", "Recall"], barmode="group")
        # fig.update_xaxes(range=[0, 1])
        # fig.update_yaxes(range=[10, limit])

        fig.update_layout(title_text="Precision/Recall at X", width=700, height=500)

        return fig

    def compute_metrics(self, threshold):
        """
        Computes standard metrics for classification models:
            -Accuracy
            -Precision
            -Recall
            -F1

        Returns the mean and standard deviation of these metrics
        over all folds in a dictionary.

        Parameters:
        threshold -- float specifying the cutoff point for
                     classification. Predictions at or above
                     the threshold are set to 1, predictions
                     below the threshold are set to 0.
        """
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        for i in range(self.K):
            
            # get all rows for fold i
            df_i = self.df_dev[self.df_dev[self.fold_col] == i]
            y_true = df_i[self.label_col]
            y_score = df_i[self.prediction_col]
            
            # apply threshold to predictions
            y_score = y_score.apply(lambda x: 1 if x >= threshold else 0)
            accuracy = accuracy_score(y_true, y_score)
            precision = precision_score(y_true, y_score)
            recall = recall_score(y_true, y_score)
            f1 = f1_score(y_true, y_score)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        dict_metrics = {}
        dict_metrics["mean_accuracy"] = round(np.mean(accuracies), 3)
        dict_metrics["sd_accuracy"] = round(np.std(accuracies), 3)
        dict_metrics["mean_precision"] = round(np.mean(precisions), 3)
        dict_metrics["sd_precision"] = round(np.std(precisions), 3)
        dict_metrics["mean_recall"] = round(np.mean(recalls), 3)
        dict_metrics["sd_recall"] = round(np.std(recalls), 3)
        dict_metrics["mean_f1"] = round(np.mean(f1s), 3)
        dict_metrics["sd_f1"] = round(np.std(f1s), 3)

        return dict_metrics
