import os, itertools, re
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import joblib
from   joblib import Parallel, delayed
from   tqdm   import tqdm

n_jobs= 20
n_jobs= min((joblib.cpu_count() - 1), n_jobs)


#TODO 
# - How do the probability distributions compare between non-overlapping contigs and sliding contigs?
    # - Either copy and edit a function to calculate windowed statistics OR edit the data so that it's offset such than each window exists (then concat)

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
        ):
    if not save_dir[-1] == '/':
        save_dir = save_dir+'/'

    dat = read_fasta(fasta_path)
    for kmer in kmers:

        print(f"Processing k-mer {kmer}")

        def _process_sequence(sub, kmer, Label, contig_lengths):
            tmp_batch = []
            sub = [sub]
            sub_len = len(sub[0][1])
            first_cols = ['Header', 'Label', 'BasePair', 'SeqLength', 'Contig']            

            def ComputeKmerIUPAC(sequence, k, as_logit = True, KmerCount = None, kmer_match_lookup = None):
                kmer_list = [sequence[i:(i+k)] for i in range(0, len(sequence)-k+1)]

                if KmerCount == None:
                    # Produce tracking dict if not provided
                    KmerCount = {''.join(e):0 for e in list(itertools.product('ACGT', repeat= k))}
                else:
                    # otherwise clear out the counts
                    KmerCount = {e:0 for e in KmerCount}
                    

                # This will be the same lookup for each contig so we can produce it for the genome and re-use it over each contig.
                if kmer_match_lookup == None:
                    # This is a clever bit of code that builds a lookup from kmers that may contain iupac codes
                    # to a list of ATCG only keys. This is done by building a regex for the unique kmers and 
                    # using said regex to identify the appropriate keys in KmerCount. This prevents us from running
                    # a regex on all of the kmers and not representing key value pairs for kmers that don't exist
                    # Also important to note is that we store the number of matches in addition to the matches.
                    # The number of matches is the denominator of the counts so we add 1/1 for one possible match
                    # and 1/3 for three.  
                    # I'm pretty sure that this could be taken care of by searching for the m
                    def _find_iupac_kmer_keys(KmerCount, inp_kmer = 'YS'):
                        iupac = {
                        'A':'A',
                        'C':'C',
                        'G':'G',
                        'T':'T',
                        'U':'T',
                        'R':'AG',
                        'Y':'CT',
                        'S':'GC',
                        'W':'AT',
                        'K':'GT',
                        'M':'AC',
                        'B':'CGT',
                        'D':'AGT',
                        'H':'ACT',
                        'V':'ACG',
                        'N':'ACGT'
                        }
                        kmer_regex = ''.join([f"[{iupac[e]}]" for e in list(inp_kmer)])
                        kmer_count_key_matches = [e for e in KmerCount.keys() if re.match(kmer_regex, e)]
                        return {inp_kmer: (kmer_count_key_matches, len(kmer_count_key_matches))}
                    # _find_iupac_kmer_keys(inp_kmer = 'YS')

                    kmer_match_lookup = {}
                    for e in set(kmer_list):
                        kmer_match_lookup |= _find_iupac_kmer_keys(KmerCount = KmerCount, inp_kmer = e)


                for kmer_i  in kmer_list:
                    keys, denominator = kmer_match_lookup[kmer_i] 
                    for key_i in keys:
                        KmerCount[key_i] += 1/denominator
                # normalize everything from likelihood to logit.
                if as_logit:
                    def logit(p):
                        if p == 1:
                            return np.inf
                        elif p == 0:
                            return -1*np.inf
                        else:
                            return np.log((p / (1-p)))

                else:
                    logit = lambda p: p
                x = len(kmer_list)
                if x <=0:
                    print(x)

                KmerCount = {k:logit(KmerCount[k]/x) for k in KmerCount}

                return KmerCount, kmer_match_lookup
        
            KmerCount, kmer_match_lookup = ComputeKmerIUPAC(sequence = sub[0][1], k = kmer, KmerCount = None, kmer_match_lookup = None)
            KmerCount |= {
                'Header': sub[0][0],
                'Label': Label, 
                'BasePair': str(0),
                'SeqLength': sub_len,
                'Contig': sub_len, 
                }
            tmp = pl.DataFrame(KmerCount)

            new_col_order = first_cols+[e for e in tmp.columns if e not in first_cols]
            tmp = tmp.select([pl.col(e) for e in new_col_order])

            tmp_batch.append(tmp) 

            for p in contig_lengths:
                if (sub_len >= p):
                    # adapted from Config_ComputeKmer
                    piece_Seq = re.findall('.{' + str(p) + '}', sub[0][1])  # Split into non-overlapping contigs of length p
                    if len(piece_Seq) == 0:
                        continue

                    tmp_lst = []
                    for contig in range(len(piece_Seq)):
                        KmerCount, kmer_match_lookup = ComputeKmerIUPAC(sequence = piece_Seq[contig], k = kmer, KmerCount = KmerCount, kmer_match_lookup = kmer_match_lookup)
                        KmerCount |= {
                            'Header': sub[0][0],
                            'Label': Label, 
                            'BasePair': str(p),
                            'SeqLength': sub_len,
                            'Contig': contig, 
                            }
                        tmp_lst.append(pl.DataFrame(KmerCount))
                    
                    tmp = pl.concat(tmp_lst)
                    new_col_order = first_cols+[e for e in tmp.columns if e not in first_cols]
                    tmp = tmp.select([pl.col(e) for e in new_col_order])
                    tmp_batch.append(tmp)

            tmp = pl.concat(tmp_batch)
            return tmp

        # out = _process_sequence(sub = sub, kmer = kmer, Label = Label, contig_lengths = contig_lengths)        
        out = Parallel(n_jobs=n_jobs)(delayed(_process_sequence)(sub = sub, kmer = kmer, Label = Label, contig_lengths = contig_lengths) for sub in tqdm(dat))
        out = pl.concat(out)

        save_file = f'{save_dir}kmer{kmer}.parquet'
        if os.path.exists(save_file):
            existing_data = pl.DataFrame(pq.read_table(save_file))
            out = pl.concat([existing_data, out])
            del existing_data
        out.write_parquet(save_file)
        del out


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
                )
        print('\n')

