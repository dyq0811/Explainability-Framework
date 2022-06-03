'''
Explainability Framework
Author: Halley Ding
Date: March 2022
'''


import shap
import scipy
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scikitplot as skplt
from math import sqrt
from scipy.stats import t
from themis_ml.metrics import *
from interpret.ext.blackbox import TabularExplainer
from raiwidgets import ExplanationDashboard


CONTINUOUS_DTYPES = [int, float]
DEFAULT_CI = 0.975


class ExpFramework:


    def __init__(self, model_path, data_path):
        self.data = pd.read_parquet(data_path, engine='pyarrow')
        self.feature_names = list(self.data.columns)
        self.model = self.load_model(model_path)
        self.explainer = TabularExplainer(self.model, self.data, features=self.feature_names)
        self.sorted_local_features, self.sorted_local_importance = self.explain_local_feature()
        # number_of_classes, number_of_samples, number_of_features
        self.n = max(10, self.sorted_local_features.shape[-1] // 20)
        self.dim = self.sorted_local_features.ndim


    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model


    def save_model(self, model, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


    def save_table(self, tab, path):
        tab.to_csv(path + '.csv', header=True, index=False)
        tab.to_parquet(path + '.parquet', engine='pyarrow')

        
    def explain_local_feature(self):
        local_explanation = self.explainer.explain_local(self.data)
        sorted_local_importance_names = np.array(local_explanation.get_ranked_local_names())
        sorted_local_importance_values = np.array(local_explanation.get_ranked_local_values())
        return sorted_local_importance_names, sorted_local_importance_values


    def explain_global_feature(self):
        global_explanation = self.explainer.explain_global(self.data)
        global_dict = global_explanation.get_feature_importance_dict()
        return np.array(list(global_dict.keys())), np.array(list(global_dict.values()))


    def importance_percentage(self, importance_matrix):
        abs_matrix = np.abs(importance_matrix)
        return abs_matrix / abs_matrix.sum(axis=-1, keepdims=True)


    def plot_top_n_features(self, feature_importance, n=5, size=(18,12)):
        feat_imp = pd.Series(np.abs(feature_importance), index=self.feature_names)
        return feat_imp.nlargest(n)[::-1].plot(kind='barh', figsize=size).figure


    def explain_interactive(self, file_path, notebook=False):
        """
        :param explanation: An object that represents an explanation.
        :type explanation: ExplanationMixin
        :param model: An object that represents a model.
            It is assumed that for the classification case
            flit has a method of predict_proba()
            returning the prediction probabilities for each
            class and for the regression case a method of predict()
            returning the prediction value.
        :type model: object
        :param dataset: A matrix of feature vector examples
            (# examples x # features),
            the same samples used to build the explanation.
            Overwrites any existing dataset on the explanation object.
            Must have fewer than 10000 rows and fewer than 1000 columns.
        :type dataset: numpy.ndarray or list[][]
        :param true_y: The true labels for the provided dataset.
            Overwrites any existing dataset on the explanation object.
        :type true_y: numpy.ndarray or list[]
        :param classes: The class names.
        :type classes: numpy.ndarray or list[]
        :param features: Feature names.
        :type features: numpy.ndarray or list[]
        :param public_ip: Optional. If running on a remote vm,
            the external public ip address of the VM.
        :type public_ip: str
        :param port: The port to use on locally hosted service.
        :type port: int
        Init docstring: Initialize the ExplanationDashboard.
        """
        if notebook:
            return ExplanationDashboard(self.explainer.explain_global(self.data), self.model, dataset=self.data)
        d = ExplanationDashboard(self.explainer.explain_global(self.data), self.model, dataset=self.data)
        html_str = d.load_index()
        html_file= open(file_path,"w", encoding='utf-8')
        html_file.write(html_str)
        html_file.close()


    def local_exp_tab_2d(self):
        out = self.data.copy()
        top_n_local = []
        top_5_exp = []
        for i in zip(self.sorted_local_features, self.sorted_local_importance):
            p1, p2, p3, p4, p5 = np.round(i[1][:5]*100/sum(np.abs(i[1])), 2)
            f1, f2, f3, f4, f5 = i[0][:5]
            template = "Top 5 features are {}({}%), {}({}%), {}({}%), {}({}%), and {}({}%)."
            top_5_exp.append(template.format(f1, p1, f2, p2, f3, p3, f4, p4, f5, p5))
            top_n_local.append(','.join(i[0][:self.n]))
        out['Top5FeaturesPct'] = top_5_exp
        out['TopFeatures'] = top_n_local
        return out


    def local_exp_tab_3d(self):
        out = self.data.copy()
        for c in range(self.sorted_local_features.shape[0]):
            top_n_local = []
            top_5_exp = []
            for i in zip(self.sorted_local_features[c], self.sorted_local_importance[c]):
                p1, p2, p3, p4, p5 = np.round(i[1][:5]*100/sum(np.abs(i[1])), 2)
                f1, f2, f3, f4, f5 = i[0][:5]
                template = "Top 5 features are {}({}%), {}({}%), {}({}%), {}({}%), and {}({}%)."
                top_5_exp.append(template.format(f1, p1, f2, p2, f3, p3, f4, p4, f5, p5))
                top_n_local.append(','.join(i[0][:self.n]))
            out['Top5FeaturesPct_class{}'.format(c+1)] = top_5_exp
            out['TopFeatures_class{}'.format(c+1)] = top_n_local
        return out


    def precision_recall(self, y_test, y_prob):
        return skplt.metrics.plot_precision_recall(y_test, y_prob)


    def roc(self, y_test, y_prob):
        return skplt.metrics.plot_roc(y_test, y_prob)


"""Utility functions for doing checks."""

def check_binary(x):
    if not is_binary(x):
        raise ValueError("%s must be a binary variable" % x)
    return x


def check_continuous(x):
    if not is_continuous(x):
        raise ValueError("%s must be a continuous variable" % x)
    return x


def is_binary(x):
    return set(x.ravel()).issubset({0, 1})


def is_continuous(x):
    return not is_binary(x) and x.dtype in CONTINUOUS_DTYPES


def s_is_needed_on_fit(estimator, s):
    if getattr(estimator, "S_ON_FIT", False):
        if s is None:
            raise ValueError(
                "Provide `s` arg when calling %s's `fit`" % estimator)
        return True
    return False


def s_is_needed_on_predict(estimator, s):
    if getattr(estimator, "S_ON_PREDICT", False):
        if s is None:
            raise ValueError(
                "Provide `s` arg when calling %s's `predict`" %
                estimator)
        return True
    return False


def mean_confidence_interval(x, confidence=0.95):
    a = np.array(x) * 1.0
    mu, se = np.mean(a), scipy.stats.sem(a)
    me = se * t._ppf((1 + confidence) / 2., len(a) - 1)
    return mu, mu - me, mu + me


"""Module for Fairness-aware scoring metrics."""

def pearson_residuals(y, pred):
    """Compute Pearson residuals.
    Reference:
    https://web.as.uky.edu/statistics/users/pbreheny/760/S11/notes/4-12.pdf
    :param array-like[int] y: target labels. 1 is positive label, 0 is negative
        label
    :param array-like[float] pred: predicted labels.
    :returns: pearson residual.
    :rtype: array-like[float]
    """
    y, pred = np.array(y), np.array(pred)
    return (y - pred) / np.sqrt(pred * (1 - pred))


def deviance_residuals(y, pred):
    """Compute Deviance residuals.
    Reference:
    https://web.as.uky.edu/statistics/users/pbreheny/760/S11/notes/4-12.pdf
    Formula:
    d = sign * sqrt(-2 * {y * log(p) + (1 - y) * log(1 - p)})
    - where sign is -1 if y = 1 and 1 if y = 0
    - y is the true label
    - p is the predicted probability
    :param array-like[int] y: target labels. 1 is positive label, 0 is negative
        label
    :param array-like[float] pred: predicted labels.
    :returns: deviance residual.
    :rtype: array-like[float]
    """
    y, pred = np.array(y), np.array(pred)
    sign = np.array([1 if y_i else -1 for y_i in y])
    return sign * np.sqrt(-2 * (y * np.log(pred) + (1 - y) * np.log(1 - pred)))


def mean_differences_ci(y, s, ci=DEFAULT_CI):
    """Calculate the mean difference and confidence interval.
    :param array-like y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param array-like s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged group and 1 is the disadvantaged
        group.
    :param float ci: % confidence interval to compute. Default: 97.5% to
        compute 95% two-sided t-statistic associated with degrees of freedom.
    :returns: mean difference between advantaged group and disadvantaged group
        with error margin.
    :rtype: tuple[float]
    """
    n0 = (s == 0).sum().astype(float)
    n1 = (s == 1).sum().astype(float)
    df = n0 + n1 - 2
    std0 = y[s == 0].std()
    std1 = y[s == 1].std()
    std_n0n1 = sqrt(((n1 - 1) * (std1) ** 2 + (n0 - 1) * (std0) ** 2) / df)
    mean_diff = y[s == 0].mean() - y[s == 1].mean()
    margin_error = t.ppf(ci, df) * std_n0n1 * \
        sqrt(1 / float(n0) + 1 / float(n1))
    return mean_diff, margin_error


def _bound_mean_difference_ci(lower_ci, upper_ci):
    """Bound mean difference and normalized mean difference CI.
    Since the plausible range of mean difference and normalized mean
    difference is [-1, 1], bound the confidence interval to this range.
    """
    lower_ci = lower_ci if lower_ci > -1 else -1
    upper_ci = upper_ci if upper_ci < 1 else 1
    return lower_ci, upper_ci


def mean_difference(y, s):
    """Compute the mean difference in y with respect to protected class s.
    In the binary target case, the mean difference metric measures the
    difference in the following conditional probabilities:
    mean_difference = p(y+ | s0) - p(y+ | s1)
    In the continuous target case, the mean difference metric measures the
    difference in the expected value of y conditioned on the protected class:
    mean_difference = E(y+ | s0) - E(y+ | s1)
    Where y+ is the desireable outcome, s0 is the advantaged group, and
    s1 is the disadvantaged group.
    Reference:
    Zliobaite, I. (2015). A survey on measuring indirect discrimination in
    machine learning. arXiv preprint arXiv:1511.00148.
    :param numpy.array y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged groupd and 1 is the disadvantaged
        group.
    :returns: mean difference between advantaged group and disadvantaged group
        with lower and uppoer confidence interval bounds.
    :rtype: tuple[float]
    """
    y = check_binary(np.array(y).astype(int))
    s = check_binary(np.array(s).astype(int))
    md, em = mean_differences_ci(y, s)
    lower_ci, upper_ci = _bound_mean_difference_ci(md - em, md + em)
    return md, lower_ci, upper_ci


def normalized_mean_difference(y, s, norm_y=None, ci=DEFAULT_CI):
    """Compute normalized mean difference in y with respect to s.
    Same the mean difference score, except the score takes into account the
    maximum possible discrimination at a given positive outcome rate. Is only
    defined when y and s are both binary variables.
    normalized_mean_difference = mean_difference / d_max
    where d_max = min( (p(y+) / p(s0)), ((p(y-) / p(s1)) )
    The d_max normalization term denotes the smaller value of either the
    ratio of positive labels and advantaged observations or the ratio of
    negative labels and disadvantaged observations.
    Therefore the normalized mean difference will report a higher score than
    mean difference in two cases:
    - if there are fewer positive examples than there are advantaged
      observations.
    - if there are fewer negative examples than there are disadvantaged
      observations.
    Reference:
    Zliobaite, I. (2015). A survey on measuring indirect discrimination in
    machine learning. arXiv preprint arXiv:1511.00148.
    :param numpy.array y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged groupd and 1 is the disadvantaged
        group.
    :param numpy.array|None norm_y: shape (n, ) or None. If provided, this
        array is used to compute the normalization factor d_max.
    :returns: mean difference between advantaged group and disadvantaged group
        with lower and upper confidence interval bounds
    :rtype: tuple(float)
    """
    y = check_binary(np.array(y).astype(int))
    s = check_binary(np.array(s).astype(int))
    norm_y = y if norm_y is None else norm_y
    d_max = float(
        min(np.mean(norm_y) / (1 - np.mean(s)),
            (1 - np.mean(norm_y)) / np.mean(s)))
    md, em = mean_differences_ci(y, s)
    # TODO: Figure out if scaling the CI bounds by d_max makes sense here.
    if d_max == 0:
        return md, md - em, md + em
    md = md / d_max
    lower_ci, upper_ci = _bound_mean_difference_ci(md - em, md + em)
    return md, lower_ci, upper_ci


def abs_mean_difference_delta(y, pred, s):
    """Compute lift in mean difference between y and pred.
    This measure represents the delta between absolute mean difference score
    in true y and predicted y. Values are in the range [0, 1] where the higher
    the value, the better. Note that this takes into account the reverse
    discrimintion case.
    :param numpy.array y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array pred: shape (n, ) containing binary predicted target,
        where 1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged groupd and 1 is the disadvantaged
        group.
    :returns: absolute difference in mean difference score between true y and
        predicted y
    :rtype: float
    """
    return abs(mean_difference(y, s)[0]) - abs(mean_difference(pred, s)[0])


def abs_normalized_mean_difference_delta(y, pred, s):
    """Compute lift in normalized mean difference between y and pred.
    This measure represents the delta between absolute normalized mean
    difference score in true y and predicted y. Values are in the range [0, 1]
    where the higher the value, the better. Note that this takes into account
    the reverse discrimintion case. Also note that the normalized mean
    difference score for predicted y's uses the true target for the
    normalization factor.
    :param numpy.array y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array pred: shape (n, ) containing binary predicted target,
        where 1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged groupd and 1 is the disadvantaged
        group.
    :returns: absolute difference in mean difference score between true y and
        predicted y
    :rtype: float
    """
    return (abs(normalized_mean_difference(y, s)[0]) -
            abs(normalized_mean_difference(pred, s)[0]))
