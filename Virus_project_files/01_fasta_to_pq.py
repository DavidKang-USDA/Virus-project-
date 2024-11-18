import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
from calculate_genotype_k_mer import Config_ComputeKmer,Genome_ComputeKmer

#TODO 
# - How do the probability distributions compare between non-overlapping contigs and sliding contigs?
    # - Either copy and edit a function to calculate windowed statistics OR edit the data so that it's offset such than each window exists (then concat)

# - Dependency issues
    # - sqlalchemy version needed for duckdb is different from the one needed for ax

# from FinalZ's train_mod.py
def read_fasta(file):
    with open(file, 'r') as f:
        sequences = []
        sequence = ''
        header = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    sequences.append((header, sequence))
                   # print(f"Appending: {header}, Sequence Length: {len(sequence)}")  # Debug print
                header = line[1:]  # Remove '>' and store the header
                sequence = ''  # Reset sequence
            else:
                sequence += line
        # Append the last read sequence
        if header:
            sequences.append((header, sequence))
           # print(f"Appending: {header}, Sequence Length: {len(sequence)}")  # Debug print
    return sequences


def fasta_to_pq(
        fasta_path = './Zea_mays',
        Label = 'Zea_mays',
        kmers = [4],
        contig_lengths = [500,1000,3000,5000,10000],
        save_dir = "./data",
        return_pr = True
        ):
    if not save_dir[-1] == '/':
        save_dir = save_dir+'/'

    dat = read_fasta(fasta_path)#[0:100] #FIXME testing is on a subset of the records
    for kmer in kmers:

        print(f"Processing k-mer {kmer}")

        tmp_batch = []  
        for sub in tqdm(dat):
            sub = [sub]
            sub_len = len(sub[0][1])
            first_cols = ['Header', 'Label', 'BasePair', 'SeqLength']

            tmp = (Genome_ComputeKmer(sub, kmer, Label, return_pr = return_pr)
                              .assign(BasePair = 'genome')
                              .assign(SeqLength = sub_len))
            tmp = tmp.loc[:, first_cols+[e for e in list(tmp) if e not in first_cols] ]
            tmp_batch.append(tmp) 

            for p in contig_lengths:
                if (sub_len >= p):
                    tmp = (Config_ComputeKmer(sub, kmer, p, Label, return_pr = return_pr)
                                      .assign(BasePair = str(p))
                                      .assign(SeqLength = sub_len))
                    tmp = tmp.loc[:, first_cols+[e for e in list(tmp) if e not in first_cols] ]
                    tmp_batch.append(tmp)
            
        tmp = pd.concat(tmp_batch)

        save_file = f'{save_dir}kmer{kmer}.parquet'
        if os.path.exists(save_file):
            existing_data = pq.read_table(save_file).to_pandas()
            tmp = existing_data.merge(tmp, how='outer')
            del existing_data

        pq.write_table(pa.Table.from_pandas(tmp), save_file)
        del tmp


fasta_paths = sorted([f"./ext_data/{e}" for e in os.listdir('./ext_data/')])

# Let's assume that all files are the species name and _might_ have a file extension (e.g. ".fasta")
for fasta_path in fasta_paths:
        species = fasta_path.split('/')[-1].split('.')[0] # species name from file name
        print(f"Processing {species}")
        print('--------------------------------------------------')
        fasta_to_pq(
                fasta_path = fasta_path,
                Label = species, 
                kmers = [i for i in range(1, 7)], # 1 - 6 # note writing many columns takes a long time. The bulk of run time seems to be writing the 6mer data
                contig_lengths = [500, 1000, 3000, 5000, 10000],
                save_dir = "./data",
                return_pr = False
                )
        print('\n')

