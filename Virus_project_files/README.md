# VirusProjectFiles 
Base on python 3

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


