import random
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import polars as pl 


# Input data
kmer = 1 # kmer doesn't matter here so we'll read in the smallest table
BasePair = 'genome' # contig length set to maximum 

# CV settings
random_seed = 89058
folds  = 5

# get basepair lengths up to 
bp = ['500', '1000', '3000', '5000', '10000', 'genome']
bp = bp[0:(1 + [i for i in range(len(bp)) if bp[i] == BasePair][0])] # add one because we want to include the end of the slice

df = pl.from_arrow(
    pq.read_table(
        f'./data/kmer{kmer}.parquet', 
        filters = [
            [
            #  ('SeqLength', '<', 10000),
             ('BasePair', 'in', bp)]
            ]
         )
) 

# check for any nulls per 
# https://stackoverflow.com/questions/78349934/polars-check-for-null-in-dataframe
assert not df.null_count().pipe(sum).item() > 0

# use Header to set up split. This will prevent two contigs from ending up in the same split
# species present
sp = (df
      .select(pl.col('Label'))
      .unique()
      .to_numpy()
      .reshape(-1,)
      .tolist()
      )

# create a mapping of species -> headers
sp_dict = {sp_i: (df
                  .filter((pl.col('Label') == sp_i))
                  .select(pl.col('Header'))
                  .unique()
                  .to_numpy()
                  .reshape(-1)
                  .tolist())
            for sp_i in sp}

sp_dict = {sp_i: sorted(sp_dict[sp_i])
            for sp_i in sp_dict}

# uniq obs for each species
sp_counts = {e:len(sp_dict[e]) for e in sorted(sp_dict.keys())}


random.seed(random_seed)
# using sorting to ensure reproduciblity conditioned on the same inputs
_ = [random.shuffle(sp_dict[sp_i]) for sp_i in  sorted(sp_dict.keys())]


# This code is not particuarly readable because I'm writing quickly.
# In brief it creates 5 fold cross validation splits w.r.t. record name (Header) with equal records
# from each species in each CV. Each fold is subdivided into validation and testing. 
# 
# To use for fold 0 for tuning
# filter Fold == fold_0_val                      -> validation Label/Headers
# filter Fold != fold_0_val & Fold != fold_0_tst ->   training Label/Headers
# Then left join with the dataset to get the relevant data.
# To use for fold 0 training
# filter Fold == fold_0_tst ->  testing Label/Headers
# filter Fold != fold_0_tst -> training Label/Headers

fold_assignments = {}
for sp_i in sorted(sp_dict.keys()):
    # break points for each fold
    # list of the start of the folds and the end
    folds_break = [i*int(sp_counts[sp_i]/folds) for i in range(folds)]+[sp_counts[sp_i]]
    # first half of a test split will be validation the rest will be true test
    # list of tuples with start of test, start of validation, end of validation and test
    folds_break = [(i, int((i+j)/2), j) for i, j in zip(folds_break, folds_break[1:])]
    folds_break

    tmp = {}
    for fold in range(folds):
        tmp |= {
                f"fold_{fold}_val" : sp_dict[sp_i][folds_break[fold][0]:folds_break[fold][1]],
                f"fold_{fold}_tst" : sp_dict[sp_i][folds_break[fold][1]:folds_break[fold][2]]
                } 
    fold_assignments |= {sp_i: tmp}


fold_split_names = sum([[f"fold_{i}_val", f"fold_{i}_tst"] for i in range(folds)], [])

fold_df = pd.concat([(pd.DataFrame(fold_assignments[sp_i][k], columns=['Header'])
                      .assign(Fold = k)
                      .assign(Label = sp_i)) for k in fold_split_names for sp_i in sp_dict.keys()])

fold_df = pl.DataFrame( fold_df.loc[:, ['Fold', 'Header', 'Label']] )


# Example usage:
# filter on Fold then join on Header/Label
# f0_hyps_val = fold_df.filter(
#     (pl.col('Fold') == 'fold_0_val')
#     ).drop(pl.col('Fold'))

# f0_hyps_val.join(df, on=['Header', 'Label'], how="left")

fold_df.write_parquet('./data/cv_folds.parquet')