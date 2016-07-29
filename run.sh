:

MKL_NUM_THREADS=8
MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=8"
MKL_DYNAMIC=FALSE

guide=guide.jpg
iter=200
scale=0.05
model=inception_3b/5x5_reduce

set -x

for input in "$@"; do

  b=$(basename $input .jpg)

  sudo -H docker run \
    --rm \
    -e MKL_NUM_THREADS=$MKL_NUM_THREADS \
    -e MKL_DOMAIN_NUM_THREADS="$MKL_DOMAIN_NUM_THREADS" \
    -e MKL_DYNAMIC=$MKL_DYNAMIC \
    -v "$PWD":/data \
    rhee/deepdream python -u /data/deepdream.py --output=$b.output --guide=$guide $input $iter $scale $model

done
