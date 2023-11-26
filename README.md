# PRIEVET
This is the source code of paper:

Edge-Protected Triangle Count Estimation under Relationship Local Differential Privacy.

# Usage
## Preprocess datasets
### Unzip Downloaded datasets
Unzip dataset files (wiki-Vote, cit-HepTh, email-Enron, and facebook) to `./dataset/[dataset-name].txt`
```
cd ./dataset
tar -xzvf [dataset-name].tar.gz
```
### Download and preprocess dataset IMDB
Download [IMDB](https://www.cise.ufl.edu/research/sparse/matrices/Pajek/IMDB.html).
Preparing IMDB with [IMDB dataset processing code](https://github.com/TriangleLDP/TriangleLDP/blob/main/README.md).
Move the processed dataset (IMDB.txt) to `./dataset/IMDB.txt`
## Run triangle counting algorithms
```
chmod -x run.sh
./run.sh
```
## Execution environment
