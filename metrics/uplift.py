import numpy as np
import pandas as pd

def get_lift_metrics(W, y, y_hat):
    """
    Standard outcome metrics for uplift models. We sort test observations
    according to the predicted treatment effect and then observe the
    treatment effect in subsets of the population.

    Args
    ----
    W : arr
        Binary indicator for treatment group.
    y : arr
        Binary indicator for outcome.
    yhat : arr
        Predicted probability for y = 1.

    Returns
    -------
    df_yhat : DataFrame
        Information about the cumulative treatment effects and related metrics.

    """
    df_yhat = pd.DataFrame(y_hat, columns=["yhat"])
    df_yhat["y"] = y.values
    df_yhat["W"] = W.values
    df_yhat = df_yhat.sort_values(by="yhat", ascending=False)
    df_yhat = df_yhat.reset_index(drop=True)
    df_yhat["cumsum_trt_obs"] = df_yhat["W"].cumsum()
    df_yhat["cumsum_ctr_obs"] = (1 - df_yhat["W"]).cumsum()
    df_yhat["cumsum_y_trt"] = (df_yhat["y"] * df_yhat["W"]).cumsum()
    df_yhat["cumsum_y_ctr"] = (df_yhat["y"] * (1 - df_yhat["W"])).cumsum()
    df_yhat["mean_y_trt"] = df_yhat["cumsum_y_trt"] / df_yhat["cumsum_trt_obs"]
    df_yhat["mean_y_ctr"] = df_yhat["cumsum_y_ctr"] / df_yhat["cumsum_ctr_obs"]
    df_yhat["cumlift"] = df_yhat["mean_y_trt"] - df_yhat["mean_y_ctr"]
    df_yhat.loc[0, "cumlift"] = 0
    df_yhat["cumlift"] = df_yhat["cumlift"].interpolate()
    df_yhat["cumlift_normal"] = df_yhat["cumlift"] / df_yhat["cumlift"].iloc[-1]

    return df_yhat