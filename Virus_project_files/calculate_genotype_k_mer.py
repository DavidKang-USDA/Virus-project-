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


def Config_ComputeKmer(in_files,kmer,p,label):
    from  collections import OrderedDict
    import itertools,re
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
    
    for line in in_files:
        tt = line.strip('\n').split(' ')
        Seq = tt[1].upper()

        piece_Seq = re.findall(r'.{'+str(p)+'}',Seq)#sequence split into non-overlapping contigs of p
        if len(piece_Seq) == 0:
            continue
        count = 1
        for piece_seq in piece_Seq:
            KmerCount = OrderedDict(zip(KmerCount.keys(),zero))
            temp = " ".join(map(str,ComputeKmerVector(piece_seq,kmer,KmerCount)))
            temp = tt[0]+'_'+str(count)+' '+temp+' '+str(label)
            IVL = temp.split(' ')
            count += 1#mark contig
            total_lst.append(IVL)
    ID_VEC_LABEL = pd.DataFrame(total_lst)
    return ID_VEC_LABEL


def Genome_ComputeKmer(in_files,kmer,label):
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
    
    for line in in_files:
        tt = line[0:-1].split(' ')
        Seq = tt[1].upper()
        KmerCount = OrderedDict(zip(KmerCount.keys(),zero))
        temp = " ".join(map(str,ComputeKmerVector(Seq,kmer,KmerCount)))
        temp = tt[0]+' '+temp+' '+str(label)
        IVL = temp.split(' ')
        total_lst.append(IVL)
    ID_VEC_LABEL = pd.DataFrame(total_lst)
    return ID_VEC_LABEL










'''#!/usr/bin/env python3
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

def queryKmer(identifier, names, kmerlen):
    from collections import OrderedDict
    import itertools
    import pandas as pd

    # Initialize Kmer dictionary (Sharing a name with a pass in variable in a for loop makes this a bit harder to read. I would recommend renaming the pass in variable kmerlen)
    KmerCount = OrderedDict((''.join(kmer), 0) for kmer in itertools.product('ACGT', repeat=kmerlen))

    total_lst = []  # This will hold all rows for the DataFrame

    names = names.split()
    print(names)
    for line in names:
        #Bug is ight here. names is a single string as passed in. Thus line is actually a character and will always have length 1. Thus you always skip this loop
    # Split the line into parts
        if len(line) < 2:  # Check if the line has at least two parts
            #print(f"Skipping incomplete line: {line}")
            continue  # Skip this line and go to the next iteration

        seq = line  # Unpack the parts into identifier and sequence
        seq.upper()  # Convert sequence to uppercase

        # Reset KmerCount to zero
        KmerCount = OrderedDict.fromkeys(KmerCount, 0)

        # Compute the k-mer vector
        kmer_vector = ComputeKmerVector(seq, kmerlen, KmerCount)
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
    
    return ID_VEC_LABEL'''

