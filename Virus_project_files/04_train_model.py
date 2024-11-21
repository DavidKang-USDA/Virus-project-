import os, json, argparse, pickle
import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

import polars as pl 

import sklearn
from   sklearn import metrics
from   sklearn.metrics import RocCurveDisplay
from   sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from   sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from   sklearn.neighbors import KNeighborsClassifier

import imblearn
from   imblearn.ensemble import BalancedBaggingClassifier

from   ax.service.ax_client import AxClient

import matplotlib.pyplot as plt



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


def append_metric(metrics_path, metric_dict):
    # load and append to stats json if it exists
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}

    stats |= metric_dict

    with open(metrics_path, 'w') as f:
        f.write(json.dumps(stats, indent=4, sort_keys=True))


def mk_plts(y, yhat, plt_path):
    cm = confusion_matrix(y, yhat)
    cm_display = ConfusionMatrixDisplay(cm)

    fpr, tpr, thresholds = metrics.roc_curve(y, yhat, pos_label=1)
    prec, recall, _ = precision_recall_curve(y, yhat, pos_label=1)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)

    # return (cm_display, roc_display, pr_display)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    cm_display.plot(ax=ax3)
    plt.savefig(plt_path)
    # plt.show()


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

# exp_json = './example.json'


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
# Optional options
parser.add_argument("--exp_json",  type = str, help="optional. path to json file containing hyperparameter specifications. If provided the name will be appended to the experiment output.")
parser.add_argument("--exp_name_append",  type = str, help="optional. ignored unless --exp_json is passed as well. appends string to experiment name instead of appending exp_json")

args = parser.parse_args()


if args.model_type:     model_type = args.model_type
if args.target_label: target_label = args.target_label
if args.kmer:                 kmer = args.kmer
if args.BasePair:         BasePair = args.BasePair
if args.cv_fold:           cv_fold = args.cv_fold
if args.cv_mode:           cv_mode = args.cv_mode
if args.tuning_iterations: tuning_iterations = args.tuning_iterations
if args.k_job:               k_job = args.k_job
if args.exp_json:         exp_json = args.exp_json
if args.exp_name_append:  exp_name_append = args.exp_name_append


#NOTE: This is where new models should be added. 
# New model_types should have at minimum:
# 1. a search space (`exp_search_space`)
# 2. an evaluation function for Ax. (`evaluate`)

match model_type:
    case 'knn':     
        from sklearn.neighbors import KNeighborsClassifier
        def train_model(
                y_train, X_train, 
                parameterization = {'weights': 'uniform', 'k': 6}, 
                k_job = 10,
                **kwargs
                ):
            model = KNeighborsClassifier(
                n_neighbors=parameterization['k'],
                weights = parameterization['weights'], 
                n_jobs= k_job)
            
            model.fit(X_train, y_train)
            return model   
        
    case 'bknn':        
        # slight modification of `evaluate()`
        def train_model(
                y_train, X_train,
                parameterization = {'weights': 'uniform', 'k': 6}, 
                k_job = 10, 
                **kwargs
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
            return model
        
    case 'rnr':
        from sklearn.neighbors import RadiusNeighborsClassifier
        def train_model(
                y_train, X_train, 
                parameterization, 
                k_job = 10
                ):
            model = RadiusNeighborsClassifier(
                radius=parameterization['radius'],
                weights = parameterization['weights'], 
                n_jobs= k_job)
            
            model.fit(X_train, y_train)
            return model

    case 'brf':
        from imblearn.ensemble import BalancedRandomForestClassifier
        def train_model(
                y_train, X_train, 
                parameterization, 
                **kwargs
                ):
            model = BalancedRandomForestClassifier(
                n_estimators=parameterization['n_estimators'], 
                max_depth = parameterization['max_depth'], 
                sampling_strategy="all", replacement=True,
                bootstrap=False
                )
            
            model.fit(X_train, y_train)
            return model

    case 'rf':
        from sklearn.ensemble import RandomForestClassifier
        def train_model(
                y_train, X_train, 
                parameterization, 
                **kwargs
                ):
            model = RandomForestClassifier(
                max_depth = parameterization['max_depth']
                )
            
            model.fit(X_train, y_train)
            return model

    case 'GNBC': # Gaussian Naive Bayes classifier
        from sklearn.naive_bayes import GaussianNB
        def train_model(
                y_train, X_train, 
                **kwargs
                ):
            model = GaussianNB()
            
            model.fit(X_train, y_train)
            return model

    case 'svml':
        from sklearn.svm import SVC
        def train_model(
                y_train, X_train, 
                parameterization, 
                **kwargs
                ):
            model = SVC(
                kernel="linear", 
                C = parameterization['C']
                )
            
            model.fit(X_train, y_train)
            return model
        
    case 'svmr':
        from sklearn.svm import SVC
        def train_model(
                y_train, X_train, 
                parameterization, 
                **kwargs
                ):
            model = SVC(
                kernel="rbf", 
                C = parameterization['C']
                )
            
            model.fit(X_train, y_train)
            return model

    case 'lr':
        from sklearn.linear_model import LogisticRegression
        def train_model(
                y_train, X_train, 
                parameterization, 
                **kwargs
                ):
            model = LogisticRegression(
                penalty=parameterization['penalty'],
                C = parameterization['C'],
                solver= 'saga' # allows l1,l2, and elasticnet
                )
            
            model.fit(X_train, y_train)
            return model
        
    case 'hgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        def train_model(
                y_train, X_train, 
                parameterization, 
                **kwargs
                ):
            model = HistGradientBoostingClassifier(
                loss = 'log_loss',
                learning_rate = parameterization['learning_rate'],
                max_iter = parameterization['max_iter'],
                max_leaf_nodes = parameterization['max_leaf_nodes'],
                max_depth = parameterization['max_depth'],
                max_features = parameterization['max_features']          
                )
            
            model.fit(X_train, y_train)
            return model

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

# Allow for arbitrary model fitting and and the addition of ad hoc text to the exp_name (for saving)
# Example of writing json from python. Expected behavior is to write from python (interactive) or manually write these files.
# with open('./example.json', 'w') as f:
#     f.write(json.dumps({"weights": "uniform", "k": 1, "n_estimators": 10}, indent=4, sort_keys=True))

# if `exp_json` exists in the global scope, load the parameters in it.
if 'exp_json' in globals():
    with open(exp_json, 'r') as f:
        params = json.load(f)

    if 'exp_name_append' in globals():
        # if `exp_name_append` was provided append this nickname to the save name.
        exp_name  = f"{model_type}-{target_label}-kmer{kmer}-bp{BasePair}-fold{cv_fold}-{exp_name_append}"
    else:
        # otherwise we'll use `exp_json` to keep everthing clear and organized.
        exp_name  = f"{model_type}-{target_label}-kmer{kmer}-bp{BasePair}-fold{cv_fold}-{exp_json.split('/')[-1]}"
else:
    # Restore ax tuner and get the best documented entry
    exp_name  = f"{model_type}-{target_label}-kmer{kmer}-bp{BasePair}-fold{cv_fold}"
    json_path = f"./models/tune/{exp_name}.json"
    ax_client = (AxClient.load_from_json_file(filepath = json_path))
    best_parameters, values = ax_client.get_best_parameters()
    params = best_parameters


# Train model ----
model = train_model(
    y_train= ys_trn, X_train= xs_trn, 
    parameterization = params, 
    k_job = k_job
)


# Save model
pickle_path = f"./models/prod/{exp_name}.pkl"

# There are more secure ways to save sklearn models than pickle. 
# This is okay for our purposes. 
# In the future migration may make sense.
with open(pickle_path, 'wb') as f:
    pickle.dump(model, f, protocol=5)

# to read a pickled model:
# with open(pickle_path, 'rb') as f:
#     model = pickle.load(f)

# Evaluate models ----
yhat_trn = model.predict(xs_trn)
yhat_tst = model.predict(xs_tst)

# Log metrics in json dictionary
# prep for auc
fpr_trn, tpr_trn, thresholds_trn = sklearn.metrics.roc_curve(ys_trn, yhat_trn, pos_label=1)
fpr_tst, tpr_tst, thresholds_tst = sklearn.metrics.roc_curve(ys_tst, yhat_tst, pos_label=1)

append_metric(
    metrics_path = f"./models/prod/{exp_name}__metrics.json", 
    metric_dict = {
        # Accuracy
        'accuracy_trn': sklearn.metrics.accuracy_score(y_true= ys_trn, y_pred= yhat_trn),
        'accuracy_tst': sklearn.metrics.accuracy_score(y_true= ys_tst, y_pred= yhat_tst),
        # Recall rate
        'recall_trn': sklearn.metrics.recall_score(y_true= ys_trn, y_pred= yhat_trn, average='binary'),
        'recall_tst': sklearn.metrics.recall_score(y_true= ys_tst, y_pred= yhat_tst, average='binary'),

        # Specificity
        'specificity_trn': sklearn.metrics.recall_score(y_true= ys_trn, y_pred= yhat_trn, average='binary', pos_label=0),
        'specificity_tst': sklearn.metrics.recall_score(y_true= ys_tst, y_pred= yhat_tst, average='binary', pos_label=0),

        # Precision
        'precision_trn': sklearn.metrics.precision_score(y_true= ys_trn, y_pred= yhat_trn, average='binary'),
        'precision_tst': sklearn.metrics.precision_score(y_true= ys_tst, y_pred= yhat_tst, average='binary'),

        # AUC
        'auc_trn': metrics.auc(fpr_trn, tpr_trn),
        'auc_tst': metrics.auc(fpr_tst, tpr_tst)
        }
    )

# Write out predictions with probabilites and convenience column for the test/train split
y_proba_trn = model.predict_proba(xs_trn)
yhat_df_trn =(
    data_trn
    .select(pl.col('Header'), pl.col('Label'))
    .with_columns( pl.Series(name="Yhat",    values=yhat_trn,      dtype=pl.Int16))
    .with_columns( pl.Series(name="ProbPos", values=y_proba_trn[:, 1], dtype=pl.Float64))
    .with_columns(pl.lit(False).alias("TestSet"))
)


y_proba_tst = model.predict_proba(xs_tst)
yhat_df_tst =(
    data_tst
    .select(pl.col('Header'), pl.col('Label'))
    .with_columns( pl.Series(name="Yhat",    values=yhat_tst,      dtype=pl.Int16))
    .with_columns( pl.Series(name="ProbPos", values=y_proba_tst[:, 1], dtype=pl.Float64))
    .with_columns(pl.lit(True).alias("TestSet"))
)

pl.concat([yhat_df_trn, yhat_df_tst]).write_parquet(f"./models/prod/{exp_name}__predictions.parquet")

# Write out visualizations
mk_plts(y = ys_trn, yhat = yhat_trn, plt_path = f"./models/prod/{exp_name}__trn.png")
mk_plts(y = ys_tst, yhat = yhat_tst, plt_path = f"./models/prod/{exp_name}__tst.png")