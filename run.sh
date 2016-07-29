:

#MKL_NUM_THREADS=8
#MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=8"
#MKL_DYNAMIC=FALSE

MKL_NUM_THREADS=4
MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=4"
MKL_DYNAMIC=FALSE

set -x

for input in "$@"; do

  b=$(basename $input .jpg)

  sudo -H docker run \
    --rm \
    -e MKL_NUM_THREADS=$MKL_NUM_THREADS \
    -e MKL_DOMAIN_NUM_THREADS="$MKL_DOMAIN_NUM_THREADS" \
    -e MKL_DYNAMIC=$MKL_DYNAMIC \
    -v "$PWD":/data \
    rhee/deepdream python -u /data/deepdream.py --output=$b.output $input 200 0.10 inception_3b/5x5_reduce

done
