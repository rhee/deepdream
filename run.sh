:

MKL_NUM_THREADS=7
MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=7"
MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS MKL_DOMAIN_NUM_THREADS MKL_DYNAMIC

iter=2400 # 100sec = 1:40sec
scale=0.05

set -e
set -x

for input in "$@"; do

  b=$(basename $input .jpg)

  python -u deepdream.py --output=$b.output $input $iter $scale $model $guide

  ./cleanup.sh "$input"

done
