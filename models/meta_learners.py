import numpy as np

class TLearner:
    """
    Simple T-Learner. The T-Learner learns subgroup treatment effects
    by training a separate model on treatment and control observations,
    then predicting a treatment and control outcome for everyone.

    Args
    ---
    treatment_learner : model
        Any classifier.
    control_learner : model
        Any classifier.

    Returns
    -------
    yhat : arr
        Predicted treatment effects.
    """

    def __init__(self, treatment_learner, control_learner):
        self.trt_learner = treatment_learner
        self.ctr_learner = control_learner

    def fit(self, data, X_names, W_name, y_name):
        data_trt = data.loc[data["condition"] == 1]
        data_ctr = data.loc[data["condition"] == 0]
        self.trt_learner.fit(data_trt[X_names], data_trt[y_name])
        self.ctr_learner.fit(data_ctr[X_names], data_ctr[y_name])

    def predict(self, data, X_names):
        yhat_trt = self.trt_learner.predict_proba(data[X_names])[:, 1]
        yhat_ctr = self.ctr_learner.predict_proba(data[X_names])[:, 1]
        return yhat_trt - yhat_ctr