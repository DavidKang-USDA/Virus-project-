import itertools
import re

options_tune = {
    'target_label': ['Drosophila_melanogaster',  'Glycine_max',  'Spodoptera_frugiperda',  'Vitis_vinifera',  'Zea_mays'],
    'BasePair': ['genome'],
    'model_type': ['bknn', 'rnr', 'brf', 'rf', 'svml', 'svmr', 'lr', 'hgb'],
    'kmer': [3],
    'fold': [0]
}


## Use itertools to set up all the combinations to run
o = options_tune

cmds = [f"python 03_tune_model.py --model_type '{e[0]}' --target_label '{e[1]}' --kmer {e[2]}  --BasePair '{e[3]}' --cv_fold {e[4]}  --cv_mode 'tuning' --k_job 30 --tuning_iterations 16"
 for e in itertools.product(
    o['model_type'],
    o['target_label'], 
    o['kmer'],
    o['BasePair'],
    o['fold'],
    )]
    


## Prepare the text of the sbatch files with informative names 
cmds = [f"""#!/bin/bash

# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --job-name='Tune {e[0]} {e[1]} cv {e[4]}'
#SBATCH --time=01:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=20   # 10 processor core(s) per node X 2 threads per core
#SBATCH --partition=short    # standard node(s)


module load apptainer
cd ../../Virus_project_files
apptainer exec ../containers/vp.sif /app/.venv/bin/python 03_tune_model.py --model_type '{e[0]}' --target_label '{e[1]}' --kmer {e[2]}  --BasePair '{e[3]}' --cv_fold {e[4]}  --cv_mode 'tuning' --k_job 30 --tuning_iterations 16"""

for e in itertools.product(
    o['model_type'],
    o['target_label'], 
    o['kmer'],
    o['BasePair'],
    o['fold'],
    )
]

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

print("""To run all the sbatch files created use:\nfor i in ./Tune*.sbatch; do echo $i; done""")