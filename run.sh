:

# NOTE: MKL thread number control
# export MKL_NUM_THREADS=4
# export MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=4"
# export MKL_DYNAMIC=FALSE

sudo -H docker run \
    --rm \
    --name deepdream \
    -e MKL_NUM_THREADS=8 \
    -e MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=8" \
    -e MKL_DYNAMIC=FALSE \
    -v "$PWD":/data \
    rhee/deepdream /deepdream.sh input.jpg 20 0.10 inception_3b/5x5_reduce
