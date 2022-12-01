"""
Description
-----------
This script generates the main results of the paper "Machine learning 
reveals how personalized climate communication can both succeed and backfire".

BibTeX
------
@inproceedings{harinen2022machine,
 title={Machine learning reveals how personalized climate communication can both succeed and backfire},
 author={Totte Harinen and Alexandre Filipowicz and Shabnam Hakimi and Rumen Iliev and Matthew Klenk and Emily Sarah Sumner},
 booktitle={NeurIPS 2022 Workshop on Causality for Real-world Impact},
 year={2022},
 url={https://openreview.net/forum?id=FAQFDIUQPpw}
}

Python version
--------------
3.8.12
"""

# Generic imports
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

import sys
import os

import random
random.seed(111)
np.random.seed(111)

# Custom imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
OUT_DIR = SCRIPT_DIR.rstrip("scripts") + "output/"
from models.meta_learners import TLearner
from metrics.uplift import get_lift_metrics
from plotting.barcharts import plot_ate_by_covariate, plot_bucket_stats

# Load data
df = pd.read_csv("https://osf.io/download/7cnpa/")

# Main effect using OLS
df["gw_hap_bin"] = np.where(df["gw_hap"] == 3, 1, 0)
mod = smf.ols(formula="gw_hap_bin ~ condition", data=df.loc[df["pre_post"] == 1])
res_main = mod.fit()

with open(OUT_DIR + "summary_main_ols.txt", "w") as fh:
    fh.write(res_main.summary().as_text())

x_tick_labs = ["Democrat", "Independent", "Republican", "Other"]
y_lab = "Believe that climate change is happening"
title = "Party"
fig_party = plot_ate_by_covariate(
    df.loc[df["pre_post"] == 1],
    "party",
    "condition",
    "gw_hap_bin",
    x_tick_labs=x_tick_labs,
    y_lab=y_lab,
    title=title,
)
fig_party.savefig(OUT_DIR + "fig_ate_by_party.png", dpi=600)

# Feature engineering
X = df.columns[:14].to_list()
X = [col for col in X if col not in ["person_ID", "condition", "pre_post"]]
df[X] = df[X].astype("category")
df_pre = df.loc[df["pre_post"] == 0].reset_index()
df_post = df.loc[df["pre_post"] == 1].reset_index()
non_features = ["condition", "gw_hap_bin"]
df_dumm = pd.get_dummies(df_post[X])
df_dumm[non_features] = df_post[non_features]

X_names = df_dumm.columns[:-2]
W_name = "condition"
y_name = "gw_hap_bin"

# Bootstrapped HTE metrics
n_iter = 1000
bucket_cate_list = []
importance_list = []
top_X_mean_list = []
bot_X_mean_list = []

for _ in range(n_iter):
    df_boot = df_dumm.sample(frac=1.0, replace=True)
    df_train_boot, df_test_boot = train_test_split(df_boot)
    treatment_learner = GradientBoostingClassifier()
    control_learner = GradientBoostingClassifier()
    tl_boot = TLearner(treatment_learner, control_learner)
    tl_boot.fit(df_train_boot, X_names, W_name, y_name)
    yhat_boot = tl_boot.predict(df_test_boot, X_names)

    importance_learner = GradientBoostingRegressor()
    importance_learner.fit(df_test_boot[X_names], yhat_boot)
    importance_list.append(importance_learner.feature_importances_)

    df_tl = get_lift_metrics(df_test_boot[W_name], df_test_boot[y_name], yhat_boot)
    df_tl["buckets"] = pd.qcut(df_tl["yhat"], 10, labels=False)

    aligned_buckets = pd.qcut(yhat_boot, 10, labels=False)
    bot_X_mean = df_test_boot.loc[aligned_buckets == 0][X_names].mean().values
    top_X_mean = df_test_boot.loc[aligned_buckets == 9][X_names].mean().values
    bot_X_mean_list.append(bot_X_mean)
    top_X_mean_list.append(top_X_mean)

    df_tl_grp = df_tl.groupby(["buckets", "W"])["y"].mean().unstack()
    bucket_cate = df_tl_grp[1] - df_tl_grp[0]
    bucket_cate_list.append(bucket_cate.values)

# Treatment effect within each bootstrapped bucket
def p5(x):
    return x.quantile(0.025)


def p95(x):
    return x.quantile(0.975)


df_cate = pd.DataFrame(bucket_cate_list).agg([np.mean, p5, p95])

plt.figure()
for idx in df_cate.T.index:
    plt.scatter(idx, df_cate.T.loc[idx, "mean"], color="red", zorder=3)
    plt.vlines(
        idx,
        df_cate.T.loc[idx, "p5"],
        df_cate.T.loc[idx, "p95"],
        color="black",
        zorder=2,
    )

plt.axhline(y=0, color="grey", linestyle="--", zorder=1)
plt.xlim(-1, 11)
plt.xlabel("Predicted treatment effect bucket (low to high)")
plt.ylabel("Observed treatment effect within bucket (95% CI)")
plt.xticks(np.arange(0, 10), np.arange(1, 11))
plt.savefig(OUT_DIR + "fig_ate_per_bucket.png", dpi=600)

# Variable importances
var_lab_dict = {
    "party": "Party",
    "race_eth": "Ethnicity",
    "age": "Age",
    "ideo": "Liberal vs conservative",
    "sex": "Gender",
    "reg_voter": "Registered voter",
    "geo": "Urban vs rural",
    "cd": "Congressional district",
    "milit": "Served in military",
    "mode": "Survey response mode",
    "evang": "Evangelical",
}

df_imp = pd.DataFrame(importance_list, columns=X_names).T
X_imp_dict = {}
for feature in X:
    X_imp_dict[feature] = np.mean(
        df_imp.loc[df_imp.index.str.contains(str(feature))].values
    )
df_imp_sorted = pd.DataFrame(X_imp_dict, index=["importance"]).T.sort_values(
    by="importance", ascending=False
)
fig, ax = plt.subplots()
sns.barplot(df_imp_sorted["importance"].values, df_imp_sorted.index, palette="Greys_r")
var_labs = [var_lab_dict[var] for var in df_imp_sorted.index]
ax.set_yticklabels(var_labs)
ax.tick_params(axis="x")
plt.tight_layout()
plt.savefig(OUT_DIR + "fig_feature_importances.png", dpi=600)

# Comparing top and bottom buckets
df_bot_X = pd.DataFrame(bot_X_mean_list, columns=X_names).agg([np.mean, p5, p95])
df_top_X = pd.DataFrame(top_X_mean_list, columns=X_names).agg([np.mean, p5, p95])

age_vars = [var for var in X_names if "age" in var]
age_labs = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
title = "Age group"

fig_age = plot_bucket_stats(df_bot_X, df_top_X, age_vars, age_labs, title)
fig_age.savefig(OUT_DIR + "fig_age_pos_neg_ate.png", dpi=600)

eth_vars = [var for var in X_names if "eth" in var]
eth_labs = [
    "White, \n non-Hispanic",
    "Hispanic",
    "Black, \n non-Hispanic",
    "Other, \n non-Hispanic",
]
title = "Ethnic group"

fig_eth = plot_bucket_stats(df_bot_X, df_top_X, eth_vars, eth_labs, title)
fig_eth.savefig(OUT_DIR + "fig_eth_pos_neg_ate.png", dpi=600)

# Population level descriptives
y_lab = "Believe that climate change is happening"
title = "Age group"
fig_age_all = plot_ate_by_covariate(
    df.loc[df["pre_post"] == 1],
    "age",
    "condition",
    "gw_hap_bin",
    age_labs,
    y_lab,
    title,
)
fig_age_all.savefig(OUT_DIR + "fig_age_ate_all.png", dpi=600)

title = "Ethnic group"
fig_eth_all = plot_ate_by_covariate(
    df.loc[df["pre_post"] == 1],
    "race_eth",
    "condition",
    "gw_hap_bin",
    eth_labs,
    y_lab,
    "Ethnic group",
)
fig_eth_all.savefig(OUT_DIR + "fig_eth_ate_all.png", dpi=600)

# Population level OLS models
mod_age = smf.ols(
    formula="gw_hap_bin ~ condition * age", data=df.loc[df["pre_post"] == 1]
)
res_age = mod_age.fit()
with open(OUT_DIR + "summary_age_ols.txt", "w") as fh:
    fh.write(res_age.summary().as_text())

mod_eth = smf.ols(
    formula="gw_hap_bin ~ condition * race_eth", data=df.loc[df["pre_post"] == 1]
)
res_eth = mod_eth.fit()
with open(OUT_DIR + "summary_eth_ols.txt", "w") as fh:
    fh.write(res_eth.summary().as_text())
