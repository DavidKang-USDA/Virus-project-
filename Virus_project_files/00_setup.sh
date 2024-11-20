#!/usr/bin/env bash

# Unzip data files discarding non-data directory
unzip virus_project_data_files.zip
rm -r __MACOSX/
# create directory for exteranal data
mkdir ext_data
mv Drosophila_melanogaster  Glycine_max  Spodoptera_frugiperda  Vitis_vinifera  Zea_mays ./ext_data/

# create directory for processed data
mkdir data
mkdir models
mkdir models/tune
mkdir models/prod
