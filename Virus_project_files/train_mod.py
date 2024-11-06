#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from calculate_genotype_k_mer import Config_ComputeKmer,Genome_ComputeKmer
import getopt,sys,os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
import os

#Command Format:- python train_mod.py --infect Drosophila_melanogaster --other Other_viruses --file mod_data --kmer 4
opts, args = getopt.getopt(sys.argv[1:], "hi:o:",["infect=","other=","file=","kmer="])

#get input parameter
for op, value in opts:
    if op == "--infect":
        infect = value
    elif op == "--other":
        other = value
    elif op == "--file":
        infile = value
    elif op == "--kmer":
        kmer = int(value)

#Determine 'infile' folder exists
isExists=os.path.exists('./'+infile+'')
if not isExists:
    os.makedirs(infile)

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

# For Drosophila_melanogaster
in_files = read_fasta(infect)

# For other viruses
oth_files = read_fasta(other)

# Example data structure: list of tuples (header, sequence)
# You need to ensure that `in_files` and `oth_files` are populated with your data as list of tuples
# Example: in_files = [('header1', 'sequence1'), ('header2', 'sequence2'), ...]

# Split data into train and test sets
train_in_files, test_in_files = train_test_split(in_files, test_size=0.2, random_state=42)
train_oth_files, test_oth_files = train_test_split(oth_files, test_size=0.2, random_state=42)

# Bootstrap the training data
bootstrapped_train_in_files = resample(train_in_files, replace=True, n_samples=len(train_in_files), random_state=1)
bootstrapped_train_oth_files = resample(train_oth_files, replace=True, n_samples=len(train_oth_files), random_state=1)

# Function to save data to a file, might be your existing function or you can define it
def save_data_to_file(data, filename):
    # Check if the directory exists, create it if not
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        for header, seq in data:
            f.write(f">{header}\n{seq}\n")


# Save the bootstrapped data and the test data to files
save_data_to_file(bootstrapped_train_in_files, './infile/bootstrapped_train_in')
save_data_to_file(bootstrapped_train_oth_files, './infile/bootstrapped_train_oth')
save_data_to_file(test_in_files, './infile/test_in')
save_data_to_file(test_oth_files, './infile/test_oth')



#In order to reduce memory usage, we only calculate 1000 sequences at a time.
step = 1000
sub_infiles = [bootstrapped_train_in_files[i:i+step] for i in range(0, len(bootstrapped_train_in_files), step)]
sub_othfiles = [bootstrapped_train_oth_files[i:i+step] for i in range(0, len(bootstrapped_train_oth_files), step)]



piece = [500,1000,3000,5000,10000]
#get Drosophila_melanogaster feature
isFirst = True
for sub in sub_infiles:
    #for make sure generate a new feature
    if isFirst:
        mode_type = 'w'
        isFirst = False
    else:
        mode_type = 'a'

    feature = Genome_ComputeKmer(sub,kmer,1)
    feature.to_csv('./'+infile+'/genome_'+str(kmer)+'_mer',mode = mode_type,sep='\t',header = None,index = False)
    #contig feature
    for p in piece:
        feature = Config_ComputeKmer(sub,kmer,p,1)
        feature.to_csv('./'+infile+'/'+str(p)+'bp_'+str(kmer)+'_mer',mode = mode_type,sep='\t',header = None,index = False)

#get other viruses feature
for sub in sub_othfiles:
    feature = Genome_ComputeKmer(sub,kmer,0)
    feature.to_csv('./'+infile+'/genome_'+str(kmer)+'_mer',mode = 'a',sep='\t',header = None,index = False)
    #contig feature
    for p in piece:
        feature = Config_ComputeKmer(sub,kmer,p,0)
        feature.to_csv('./'+infile+'/'+str(p)+'bp_'+str(kmer)+'_mer',mode = 'a',sep='\t',header = None,index = False)


#clear useless variable, Free memory
del feature,sub_infiles,sub_othfiles,in_files,oth_files


# Read the content of the two text files
with open('./infile/test_in', 'r') as f1, open('./infile/test_oth', 'r') as f2:
    content1 = f1.read()
    content2 = f2.read()

# Combine the content
combined_content = content1 + '\n' + content2

# Save the combined content to a new text file
with open('./infile/test_file', 'w') as f:
    f.write(combined_content)
