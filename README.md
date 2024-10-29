This repository provides supporting information and the codes for the following project<br />
 <br />

## Create a machine learning tool to predict virus cross-infection risk in plants and insects.


Dependencies
-----------
Python packages "pandas","numpy","sklearn","imblearn" are needed to be installed before Installation of train_mod.py and predResult.py
human-infecting virus finder is tested to work under Python 3.6+, the dependency requirements release:

* scipy( >= 1.2)
* numpy( >= 0.16)
* scikit-learn( >= 0.21)
* pandas( >= 0.24)
* imbalanced-learn( >= 0.4.3)
* joblib( >= 0.13)


Installation
-----------
To install "pandas", "numpy" and "sklearn", open terminal and input,

	conda install pandas
	or
	pip install pandas

	conda install numpy
	or
	pip install numpy

	conda install scikit-learn
	or
	pip install -U scikit-learn

To install "imblearn", open terminal and input,

	conda install -c glemaitre imbalanced-learn
	or
	pip install -U imbalanced-learn


Usage
-----------
### decompression the virus_project_data_files.zip

	unzip virus_project_data_files.zip

### example command

	python train_mod.py --infect Drosophila_melanogaster --other Glycine_max --file mod_data --kmer 4


Arguments
-----------
--infect  The file of the virus sequences.

--other The file of the other viruses sequences.

--file data The folder of saving generate viral contig k-mer.

--kmer  The length of the k-mer.


Description
-----------
Before running the train_mod.py, please download and extract the virus_project_data_files.zip to the current directory at first. 

The viral genomes containing virus and other viruses were split into non-overlapping contigs of 500, 1,000, 3,000, 5,000 and 10,000 nucleotides long.



Usage
-----------
### example command

	python predResult.py --query "./infile/test_file" --output predict_result --file mod_data --kmer 4 --kjob 1 --bjob 1

Arguments
-----------
--query The file of query sequences.

--output  The name of the predict result file.

--file data The folder of using viral contig k-mer.

--kmer  The length of the k-mer.

--kjob  The number of parallel jobs to run KNeighborsClassifier.

--bjob  The number of parallel jobs to run BalancedBaggingClassifier.

Description
-----------
The function identifies host-infecting virus contig in the input file using the model trained based on the viral genomes.

The rows correspond to sequences, and the columns are from the left to the right, sequence name (Name), using model (Model), prediction result (Label) and prediction probability (Probability).

1- When the prediction result is "1" and prediction probability is close to 1, it means the model predicts this query sequence more likely to infect (in virus).

2- When the prediction result is "0" and prediction probability is close to 0, it means the model predicts this query sequence more likely to infect (oth virus).

For a query sequence of length L: if L<1kb, the model trained by 500bp sequences is used to predict; if 1kb<=L<3kb, the model trained by 1000bp sequences is used to predict; if 3kb<=L<5kb, the model trained by 3000bp sequences is used to predict; if 5kb<=L<10kb, the model trained by 5000bp sequences is used by predict; if 10kb<=L<15kb, the model trained by 10000bp sequences is used to predict; if L>=15kb, the model trained by the viral genome is used to predict.

The different between kjob and bjob: bjob will faster the kjob, but it will take more memory when using same processors.
