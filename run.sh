:

set -x

for input in *.jpg; do

  rm -fr output/tmp/*

  sudo -H docker run \
    --rm \
    --name deepdream \
    -e MKL_NUM_THREADS=8 \
    -e MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=8" \
    -e MKL_DYNAMIC=FALSE \
    -v "$PWD":/data \
    rhee/deepdream /deepdream.sh $input 100 0.10 inception_3b/5x5_reduce

  b=$(basename $input .jpg)
  mkdir -p $b.output
  mv output/*.jpg $b.output/

done
