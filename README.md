# PRIEVET
This is the source code of paper:

Edge-Protected Triangle Count Estimation under Relationship Local Differential Privacy.

# Usage
## Preprocess datasets
### Unzip Downloaded datasets
Unzip dataset files (wiki-Vote, cit-HepTh, email-Enron, and facebook) to `./dataset/[dataset-name].txt`
```shell
cd ./dataset
tar -xzvf [dataset-name].tar.gz
```
### Download and preprocess dataset IMDB
1. Download [IMDB](https://www.cise.ufl.edu/research/sparse/matrices/Pajek/IMDB.html).
2. Preparing IMDB with [IMDB dataset processing code](https://github.com/TriangleLDP/TriangleLDP/blob/main/README.md).
3. Move the processed dataset (IMDB.txt) to `./dataset/IMDB.txt`
## Run triangle counting algorithms
```
chmod +x run.sh
./run.sh [dataset-name (wiki-Vote/facebook/cit-HepTh/email-Enron/IMDB)]
```
## Execution environment
We used python 3.8.5 on Ubuntu 20.04.3 LTS.
