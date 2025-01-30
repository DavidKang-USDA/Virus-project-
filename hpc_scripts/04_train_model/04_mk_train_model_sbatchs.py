import itertools
import re

options_tune = {
    'target_label': ['Drosophila_melanogaster',  'Glycine_max',  'Spodoptera_frugiperda',  'Vitis_vinifera',  'Zea_mays'],
    'BasePair': ['genome'],
    'model_type': ['bknn', 'rnr', 'brf', 'rf', 'svml', 'svmr', 'lr', 'hgb'],
    'kmer': [3],
    'fold': [0]
}
options_train = options_tune.copy()
options_train['fold'] = [i for i in range(5)]
options_train['model_type'].append('GNBC')


## Use itertools to set up all the combinations to run
o = options_train

opts = [e for e in itertools.product(
    o['model_type'],
    o['target_label'], 
    o['kmer'],
    o['BasePair'],
    o['fold'],
    )]


## Prepare the text of the sbatch files with informative names 
cmds = []
for e in opts:
    sbatch_txt = f"""#!/bin/bash

# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --job-name='Train {e[0]} {e[1]} cv {e[4]}'
#SBATCH --time=01:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=20   # 10 processor core(s) per node X 2 threads per core
#SBATCH --partition=short    # standard node(s)


module load apptainer
cd ../../Virus_project_files"""

    # Training logic is a little more complex than tuning logic:
    sbatch_add = []
    if e[0] == 'GNBC':
        sbatch_add.append([
            "# If training GNBC, hyperparameters shouldn't be needed. Write an empty file to match expectation of hyps",
            f"cp ./models/tune/bknn-Drosophila_melanogaster-kmer3-bpgenome-fold0.json ./models/tune/{e[0]}-{e[1]}-kmer{e[2]}-bp{e[3]}-fold{e[4]}.json"    
        ])
    else:
        if e[4] != 0:
            sbatch_add.append([
                "# copy the hyperparameter tuning file with a new name so that the folds name matches",
                f"cp ./models/tune/{e[0]}-{e[1]}-kmer{e[2]}-bp{e[3]}-fold0.json ./models/tune/{e[0]}-{e[1]}-kmer{e[2]}-bp{e[3]}-fold{e[4]}.json"
                ])
            
    sbatch_add.append([
        "# Run tuning",
        f"apptainer exec ../containers/vp.sif /app/.venv/bin/python 04_train_model.py --model_type '{e[0]}' --target_label '{e[1]}' --kmer {e[2]}  --BasePair '{e[3]}' --cv_fold {e[4]}  --cv_mode 'training' --k_job 20"
    ])

    if ((e[0] == 'GNBC') | (e[4] != 0)):
        sbatch_add.append([
            "#Non-0 folds use a copied tuning json. (or created a phony one for GNBC) Cleanup the tuning dir by deleting this. ",
            f"rm ./models/tune/{e[0]}-{e[1]}-kmer{e[2]}-bp{e[3]}-fold{e[4]}.json"
        ])

    sbatch_add = '\n'.join(sum(sbatch_add, []))
    sbatch_txt = sbatch_txt+'\n'+sbatch_add
    cmds.append(sbatch_txt)


## Write out sbatch files and instructions on use
print('Writing:')
for e in cmds:
    new_name = (
        re.findall(r'job-name=.+\n', e)[0]
        .replace('\n', '')
        .replace('job-name=', '')
        .replace('\'', '')
        .replace(' ', '_')
        )
    new_name = f'./{new_name}.sbatch'
    print(f'\t{new_name}') 
    with open(new_name, 'w') as f:
        f.writelines(e)

print("""To run all the sbatch files created use:\nfor i in ./Train*.sbatch; do echo $i; done""")