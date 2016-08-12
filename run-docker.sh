#!/bin/sh

CAFFE_HOME=/caffe
CUDA_ENABLED=

MKL_NUM_THREADS=7
MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=7"
MKL_DYNAMIC=FALSE

if [ -z "$1" ]; then
  exec sudo -H docker run \
      --name=dream-notebook \
      --rm \
      -p 8888:8888 \
      -v "$PWD":/data \
      -e CAFFE_HOME=$CAFFE_HOME \
      -e MKL_NUM_THREADS=$MKL_NUM_THREADS \
      -e MKL_DOMAIN_NUM_THREADS="$MKL_DOMAIN_NUM_THREADS" \
      -e MKL_DYNAMIC=$MKL_DYNAMIC \
      rhee/deepdream jupyter notebook --ip=0.0.0.0 --no-browser
else
  exec sudo -H docker run \
      --name=dream \
      --rm \
      -t -i \
      -p 8888:8888 \
      -v "$PWD":/data \
      -e CAFFE_HOME=$CAFFE_HOME \
      -e MKL_NUM_THREADS=$MKL_NUM_THREADS \
      -e MKL_DOMAIN_NUM_THREADS="$MKL_DOMAIN_NUM_THREADS" \
      -e MKL_DYNAMIC=$MKL_DYNAMIC \
      rhee/deepdream "$@"
fi
