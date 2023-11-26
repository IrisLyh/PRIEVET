if [ $# -ne 1 ]; then
    echo "USAGE: run_TriangleCounting.sh [Dataset]"
    exit 1
fi

python3 main.py --dataset ${1} --epsilon 1 --scheme CaliToUpper
python3 main.py --dataset ${1} --epsilon 1 --scheme CaliToLS
python3 main.py --dataset ${1} --epsilon 1 --scheme CaliToTrunc
python3 main.py --dataset ${1} --epsilon 6 --scheme CaliToUpper
python3 main.py --dataset ${1} --epsilon 6 --scheme CaliToLS
python3 main.py --dataset ${1} --epsilon 6 --scheme CaliToTrunc
