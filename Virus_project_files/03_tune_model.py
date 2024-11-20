import os, argparse
import numpy as np
import pandas as pd
import polars as pl 
import pyarrow as pa
import pyarrow.parquet as pq

import sklearn
from   sklearn.neighbors import KNeighborsClassifier

import imblearn
from   imblearn.ensemble import BalancedBaggingClassifier

from   ax.service.ax_client import AxClient, ObjectiveProperties
# from ax.utils.notebook.plotting import init_notebook_plotting, render



def mk_np_from_pq(data):
    # confirm no nulls
    assert not data.null_count().pipe(sum).item() > 0
    ys = data.select(pl.col('Label')).to_numpy().reshape(-1)

    xs = data.drop(
        pl.col('Header'),
        pl.col('Label'), 
        pl.col('BasePair'), 
        pl.col('SeqLength'),
        pl.col('Contig')
        ).to_numpy()
    
    # convert logits to probability
    xs = np.exp(xs)/(1+np.exp(xs))
    return((ys, xs))

def mk_y_onehot(ys, target_label):
    y_onehot = np.zeros_like(ys)
    y_onehot[ys == target_label] = 1
    y_onehot = y_onehot.astype(int)
    return y_onehot


# default values ----
# (overwritten by parser)
# Model options
model_type = 'knn'
target_label = 'Glycine_max'
# Data options
kmer = 1 
BasePair = '3000'
cv_fold = 0 
cv_mode = 'tuning'
# Tuner options
tuning_iterations = 20
# Evaluation options
k_job = 10

# argparse values ----
parser = argparse.ArgumentParser()
# Model options
parser.add_argument("--model_type",   type = str, help="model to be used: knn")
parser.add_argument("--target_label", type = str, help="prediction target: 'Drosophila_melanogaster',  'Glycine_max',  'Spodoptera_frugiperda', 'Vitis_vinifera', 'Zea_mays'")
# Data options
parser.add_argument("--kmer",     type = int, help="kmer data to be used: 1-6")
parser.add_argument("--BasePair", type = str, help="contig length: '500', '1000', '3000', '5000', '10000', 'genome' ")
parser.add_argument("--cv_fold",  type = int, help="cv fold to be used: 0-4")
parser.add_argument("--cv_mode",  type = str, help="cv mode to be used: 'tuning' or 'training'")
# Tuner options
parser.add_argument("--tuning_iterations", type = int, help="number of trials to be run by the hyperparameter tuner.")
# Evaluation options
parser.add_argument("--k_job",             type = int, help="number of jobs for knn and bagging classifier") 

args = parser.parse_args()


if args.model_type:     model_type = args.model_type
if args.target_label: target_label = args.target_label
if args.kmer:                 kmer = args.kmer
if args.BasePair:         BasePair = args.BasePair
if args.cv_fold:           cv_fold = args.cv_fold
if args.cv_mode:           cv_mode = args.cv_mode
if args.tuning_iterations: tuning_iterations = args.tuning_iterations
if args.k_job:               k_job = args.k_job




#NOTE: This is where new models should be added. 
# New model_types should have at minimum:
# 1. a search space (`exp_search_space`)
# 2. an evaluation function for Ax. (`evaluate`)

match model_type:
    case 'knn':        
        exp_search_space = [
            {
                "name": "weights",
                "type": "choice",
                "values": ['uniform', 'distance'], 
                "is_ordered": True,
                "sort_values": False,
                "value_type": "str"
            },
            {
                "name": "k",
                "type": "range",
                "bounds": [1, 10],
                "value_type": "int", 
                "log_scale": False,  
            },
            {
                "name": "n_estimators",
                "type": "range",
                "bounds": [1, 10],
                "value_type": "int", 
                "log_scale": False,  
            }
        ]
        # define evaluation function for Ax to use. 
        # NOTE: this is written following the approach used by _Zhang et al. 2019_ of using a random seed to set up a test/train split.
        # I (Daniel) think it would be better to use a test/validation/training split and make definition of these _explicit_ even if `train_test_split` is generating these.  
        def evaluate(
                y_train, X_train, 
                y_test,  X_test,
                parameterization = {'weights': 'uniform', 'k': 6}, 
                k_job = 10
                ):
            knn = KNeighborsClassifier(
                n_neighbors=parameterization['k'],
                weights = parameterization['weights'], 
                n_jobs= k_job)

            model = BalancedBaggingClassifier(
                estimator=knn, 
                n_estimators =parameterization['n_estimators'], 
                n_jobs = k_job)
            
            model.fit(X_train, y_train)
            return {'score': ( model.score(X_test, y_test) )}
        
    case _:
        print(f"Model {model_type} is not defined!")
        assert True == False


# Data prep ----
## Prepare filters for the train/test (or train/validation) splits
fold_df = pl.from_arrow(pq.read_table('./data/cv_folds.parquet'))

match cv_mode:
    case 'tuning':
        mask_trn = ((pl.col('Fold') != f"fold_{cv_fold}_val") & 
                    (pl.col('Fold') != f"fold_{cv_fold}_tst"))
        
        mask_tst = ((pl.col('Fold') == f"fold_{cv_fold}_val"))

    case 'training':
        mask_trn = ((pl.col('Fold') != f"fold_{cv_fold}_tst"))
        
        mask_tst = ((pl.col('Fold') == f"fold_{cv_fold}_tst"))

## Load in data for target kmer
# Basically this is a convoluted way to access parquet data as a np array
# converting directly fails because we can't take a column as a pa.array and convert that. 
# get basepair lengths up to target bp
bp = ['500', '1000', '3000', '5000', '10000', 'genome']
bp = bp[0:(1 + [i for i in range(len(bp)) if bp[i] == BasePair][0])] # add one because we want to include the end of the slice

df = pl.from_arrow(pq.read_table(f'./data/kmer{kmer}.parquet', 
                                 filters = [[('BasePair', 'in', bp)]])
                                 ) 

# use inner join so that nans are not introduced for cases where a record is not present at all bp
data_trn = fold_df.filter(mask_trn).drop(pl.col('Fold')).join(df, on=['Header', 'Label'], how="inner")
data_tst = fold_df.filter(mask_tst).drop(pl.col('Fold')).join(df, on=['Header', 'Label'], how="inner") 

ys_trn, xs_trn = mk_np_from_pq(data = data_trn)
ys_tst, xs_tst = mk_np_from_pq(data = data_tst)

ys_trn = mk_y_onehot(ys = ys_trn, target_label = target_label)
ys_tst = mk_y_onehot(ys = ys_tst, target_label = target_label)

# Restore or create ax tuner
exp_name  = f"{model_type}-{target_label}-kmer{kmer}-bp{BasePair}-fold{cv_fold}"
json_path = f"./models/tune/{exp_name}.json"
if not os.path.exists(json_path):
    ax_client = AxClient()
    ax_client.create_experiment(
        name=exp_name,
        parameters=exp_search_space,
        objectives={"score": ObjectiveProperties(minimize=False)} # Score is mean accuracy so we want to maximize it
    )
else: 
    ax_client = (AxClient.load_from_json_file(filepath = json_path))


# NOTE: The exception below is also present in the example documents for Ax. 
# Encountered exception in computing model fit quality: RandomModelBridge does not support prediction.
for i in range(tuning_iterations):
    params, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(
        y_train= ys_trn, X_train= xs_trn, 
        y_test= ys_tst, X_test= xs_tst,
        parameterization = params, 
        k_job = k_job
        ))

ax_client.save_to_json_file(filepath = json_path)

# if working interactively in a jupyter notebook here are some useful methods. 
# ax_client.get_trials_data_frame()
# best_parameters, values = ax_client.get_best_parameters()
#
#  # This should be in its own cell
# init_notebook_plotting()
#
# # next cell
# render(ax_client.get_optimization_trace(objective_optimum=1))