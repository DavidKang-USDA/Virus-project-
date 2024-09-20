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
def ComputeKmerVector(sequence,k,KmerCount):
    from  collections import OrderedDict
    n = len(sequence)
    for j in range(0,n-k+1):
        kmer = sequence[j:(j+k)]
        if kmer in KmerCount:
            KmerCount[kmer] += 1
        else:
            continue

    KmerVec = OrderedDict()
    for m in KmerCount:
        KmerVec[m] = float(KmerCount[m])/float(n-k+1)
    Vec = list(KmerVec.values())
    return Vec

def queryKmer(name,kmer):
    from  collections import OrderedDict
    import itertools
    import pandas as pd
    
    KmerCount = OrderedDict()
    s = itertools.product('ACGT',repeat = kmer)
    sl = list(s)
    for i in range(0,len(sl)):
        a0 = str(sl[i])
        a1 = a0.replace("(","")
        a2 = a1.replace(")","")
        a3 = a2.replace(",","")
        a4 = a3.replace("'","")
        a4 = a4.replace(" ", "")
        KmerCount[a4] = 0
    zero = [0 for _ in range(len(sl))]
    
    total_lst = list()
    for line in name:
        tt = line.strip('\n').split('\n')
        #print(tt)
        #print(tt[1].upper())
        Seq = tt[1].upper()
        KmerCount = OrderedDict(zip(KmerCount.keys(),zero))
        temp = "        ".join(map(str,ComputeKmerVector(Seq,kmer,KmerCount)))
        temp = tt[0]+'      '+temp
        #
        #print(temp)
        IVL = temp.split('      ')
        #print(IVL)
        total_lst.append(IVL)
    ID_VEC_LABEL = pd.DataFrame(total_lst)
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







