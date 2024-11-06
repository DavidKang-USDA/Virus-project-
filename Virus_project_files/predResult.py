#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from calculate_genotype_k_mer import queryKmer
import pandas as pd
import imblearn
# print(imblearn.__version__)
from imblearn.ensemble import BalancedBaggingClassifier
# help(BalancedBaggingClassifier)
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import getopt,sys
import time

start_time = time.time()


#python predResult.py --query test --output predict_result --file mod_data --kmer 4 --kjob 1 --bjob 1
opts, args = getopt.getopt(sys.argv[1:], "hi:o:",["query=","output=","file=","kmer=","kjob=","bjob="])
for op, value in opts:
    if op == "--query":
        query = value
    elif op == "--output":
        output = value
    elif op == "--file":
        infile = value
    elif op == "--kmer":
        kmer = int(value)
    elif op == "--kjob":
        k_job = int(value)
    elif op == "--bjob":
        b_job = int(value)

###########################################################
#classifier query sequences by different length
# test_500, test_1000,test_3000,test_5000,test_10000,test_genome = [],[],[],[],[],[]
# Initialization of lists for different sequence lengths
test_500 = []
test_1000 = []
test_3000 = []
test_5000 = []
test_10000 = []
test_genome = []


#file = open(query,'r')
query_name = []
#sequences = parse_fasta(query)

name = ""  # Default or initial value for 'name'
file = open(query, 'r')
for line in file:
    line = line.strip(' ')
    if line.startswith('>'):  # Condition to check and set 'name'
        name = line
        #print(name)
        continue
    if len(line) < 1000:
        test_500.append(name+' '+line)
    elif len(line) < 3000:
        test_1000.append(name+' '+line)
    elif len(line) < 5000:
        test_3000.append(name+' '+line)
    elif len(line) < 10000:
        test_5000.append(name+' '+line)
    elif len(line) < 15000:
        test_10000.append(name+' '+line)
    else:
        test_genome.append(name+' '+line)
file.close()

result = pd.DataFrame({"name":query_name})
result.index = result.iloc[:,0]

###########################################################
lengths = [test_500,test_1000,test_3000,test_5000,test_10000,test_genome]
piece = ['500bp','1000bp','3000bp','5000bp','10000bp','genome']
pred_result = []
isFirst = True
for ll in range(len(lengths)):
    if lengths[ll] == []:
        continue
    #print(lengths)
########################################
    data = pd.read_csv('./'+infile+'/'+piece[ll]+'_'+str(kmer)+'_mer',sep = '\t',header = None,index_col = False)
    X = data.iloc[:,1:(data.shape[1] - 1)]#get fearture
    y = data.iloc[:,data.shape[1] - 1]#get label
    #print(y)
    #del data
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(len(X_train))
    del X,y#clear useless variable, Free memory
    
    #Base Estimater
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=k_job)
    #set model parameter
    model = BalancedBaggingClassifier(estimator=knn, n_estimators =10, n_jobs = b_job)
    #training the model
    model.fit(X_train,y_train)
    
    #clear useless variable, Free memory
    del X_train,y_train
    
    print(''+piece[ll]+' model ready')
#########################################
#predict query sequences

    #In order to reduce memory usage, we only calculate 1000 sequences at a time.
    step = 1000
    #print(lengths[ll])
    sub_query = [lengths[ll][i:i + step] for i in range(0, len(lengths[ll]), step)]
    #print(sub_query)
    # Filter out sequences that are too short before processing
    # Ensure sub_query is defined and populated correctly first
    # sub_query = [' '.join(s.split(' ')[0:2]) for s in sub_query for sub in s if len(sub.split(' ')[1]) >= kmer]
    #flat_sub_query = [item for sublist in sub_query for item in sublist]

    # Now, filter out sequences that are too short to calculate k-mers
    #filtered_sub_query = [entry for entry in flat_sub_query if ' ' in entry and len(entry.split(' ')) >= 2]
    #print(sub_query)
    # X_test = np.empty(256) # original
     #In order to reduce memory usage, we only calculate 1000 sequences at a time.
    #step = 1000
    #sub_query = [lengths[ll][i:i+step] for i in range(0,len(lengths[ll]),step)]
    #for sub in sub_query:
        #data = queryKmer(sub,kmer)
    X_test = 0
    for sub in sub_query:
        #print(sub)
        data = queryKmer(sub, kmer) 
        #get query sequence fearture
        #X_test = np.vstack([X_test,np.array(data.iloc[:,1:])]) # origina
        X_test = np.array(data.iloc[:,1:])
        #print(X_test)
        #print(X_test.shape)
        y_pred = np.array(model.predict(X_test))
        #print(len(y_pred))
        predict_prob_y = np.array(model.predict_proba(X_test)[:,1])
        pred_data = pd.DataFrame(
            {
            'Model' : piece[ll],
            'Label' : y_pred,
            'Probability' : predict_prob_y
            }
            
        )
        pred_data.index = data.iloc[:,0]
        #print(pred_data.index)
        if isFirst:
            isFirst = False
            pred_result = pred_data
        else:#append predict result
            pred_result = pred_result._append(pred_data)

        #clear useless variable, Free memory
        #del data, X_test
        print("Data shape:", X_test.shape)
        y_pred = model.predict(X_test)
        predict_prob_y = model.predict_proba(X_test)[:, 1]  # assuming binary classification

        print("Sample predictions:", y_pred[:5])
        print("Sample probabilities:", predict_prob_y[:5])

        # Ensure DataFrame is being updated
        pred_data['Label'] = y_pred
        pred_data['Probability'] = predict_prob_y
        #print("Preview of prediction DataFrame:", pred_data.head())
    del data, X_test
    
#print the predict result
result = result.join(pred_result, how='outer')
result.drop("name",axis=1, inplace=True)
result.index.name = 'Name'
result.to_csv(''+output+'',sep = '\t',header = True,index = True)

end_time = time.time()
total_time = end_time - start_time
print ('Total running time of the program: %.2f seconds' % (total_time))
#clear useless variable, Free memory
del pred_result, result,test_500,test_1000,test_3000,test_5000,test_10000,test_genome,lengths,model
