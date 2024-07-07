#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:50:53 2019

@author: FinalZ
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 21:03:07 2017

@author: zhang
"""
def ComputeKmerVector(seq, kmer_size, kmer_count):
    n = len(seq)
    if n < kmer_size:  # Check if the sequence is too short to form even one k-mer
        print(f"Skipping a sequence because it is too short: {seq}")  # Debugging output
        return [0] * len(kmer_count)  # Return a vector of zeros or handle as appropriate

    # Reset the kmer counts for safety
    for key in kmer_count.keys():
        kmer_count[key] = 0

    # Calculate k-mer frequencies
    for i in range(n - kmer_size + 1):
        kmer = seq[i:i + kmer_size]
        if kmer in kmer_count:
            kmer_count[kmer] += 1

    # Compute k-mer vector, ensuring we do not divide by zero
    total_possible_kmers = n - kmer_size + 1
    kmer_vector = [float(kmer_count[kmer]) / total_possible_kmers for kmer in kmer_count]
    return kmer_vector

def queryKmer(names, kmer):
    from collections import OrderedDict
    import itertools
    import pandas as pd

    # Initialize Kmer dictionary
    KmerCount = OrderedDict((''.join(kmer), 0) for kmer in itertools.product('ACGT', repeat=kmer))

    total_lst = []  # This will hold all rows for the DataFrame

    for line in names:
        parts = line.strip().split(' ', 1)  # Split the line into parts
        if len(parts) < 2:  # Check if the line has at least two parts
            print(f"Skipping incomplete line: {line}")
            continue  # Skip this line and go to the next iteration

        identifier, seq = parts  # Unpack the parts into identifier and sequence
        seq = seq.upper()  # Convert sequence to uppercase

        # Reset KmerCount to zero
        KmerCount = OrderedDict.fromkeys(KmerCount, 0)

        # Compute the k-mer vector
        kmer_vector = ComputeKmerVector(seq, kmer, KmerCount)
        if kmer_vector is None:  # Check if the k-mer vector computation was successful
            print(f"Unable to compute k-mer vector for the sequence: {seq}")
            continue

        # Prepare the row for the DataFrame
        row = [identifier] + kmer_vector
        total_lst.append(row)

    if not total_lst:  # Check if no data was added to total_lst
        print("No valid data was processed.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Create DataFrame from total_lst
    column_names = ['Identifier'] + list(KmerCount.keys())
    ID_VEC_LABEL = pd.DataFrame(total_lst, columns=column_names)

    return ID_VEC_LABEL


def Config_ComputeKmer(in_files, kmer, p, label):
    from collections import OrderedDict
    import itertools, re
    import pandas as pd
    
    # Initialize Kmer dictionary
    KmerCount = OrderedDict((''.join(kmer), 0) for kmer in itertools.product('ACGT', repeat=kmer))

    # Placeholder for total results
    total_lst = []

    for header, seq in in_files:  # Unpack the header and sequence from each tuple
        Seq = seq.upper()  # Ensure the sequence is in uppercase
        piece_Seq = re.findall('.{' + str(p) + '}', Seq)  # Split into non-overlapping contigs of length p

        if len(piece_Seq) == 0:
            continue

        count = 1
        for piece_seq in piece_Seq:
            # Reset the KmerCount for each sequence piece
            KmerCount = OrderedDict.fromkeys(KmerCount, 0)

            # Compute the kmer vector for the sequence piece
            kmer_vector = ComputeKmerVector(piece_seq, kmer, KmerCount)

            # Create a list entry combining header, kmer counts and label
            entry = [header + '_' + str(count)] + kmer_vector + [label]
            
            # Increment the piece count
            count += 1
            
            # Append the list to the total list
            total_lst.append(entry)
    
    # Convert list of lists to DataFrame
    columns = ['Header'] + list(KmerCount.keys()) + ['Label']
    ID_VEC_LABEL = pd.DataFrame(total_lst, columns=columns)
    
    return ID_VEC_LABEL



def Genome_ComputeKmer(in_files, kmer, label):
    from collections import OrderedDict
    import itertools
    import pandas as pd
    
    # Initialize Kmer dictionary
    KmerCount = OrderedDict((''.join(kmer), 0) for kmer in itertools.product('ACGT', repeat=kmer))

    # Placeholder for total results
    total_lst = []
    
    for header, seq in in_files:  # Unpack the header and sequence from each tuple
        Seq = seq.upper()  # Ensure the sequence is in uppercase
        # Reset the KmerCount for each sequence
        KmerCount = OrderedDict.fromkeys(KmerCount, 0)
        
        # Compute the kmer vector for the sequence
        kmer_vector = ComputeKmerVector(Seq, kmer, KmerCount)
        # Create a list entry combining header, kmer counts and label
        entry = [header] + kmer_vector + [label]
        
        # Append the list to the total list
        total_lst.append(entry)
    
    # Convert list of lists to DataFrame
    ID_VEC_LABEL = pd.DataFrame(total_lst, columns=['Header'] + list(KmerCount.keys()) + ['Label'])
    
    return ID_VEC_LABEL

