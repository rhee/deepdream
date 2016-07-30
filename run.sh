:

MKL_NUM_THREADS=8
MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=8"
MKL_DYNAMIC=FALSE

# lower level than the defualt inception_4c/output,
# gets more patternish output
#model=--model=inception_3b/5x5_reduce
model=--model=inception_4a/output

# guide image binds the result, not an interesting
#guide=--guide=guide.jpg

iter=2400 # 100sec = 1:40sec
scale=0.05

set -e
set -x

for input in "$@"; do

  b=$(basename $input .jpg)

  sudo --set-home docker run \
    --rm \
    -e MKL_NUM_THREADS=$MKL_NUM_THREADS \
    -e MKL_DOMAIN_NUM_THREADS="$MKL_DOMAIN_NUM_THREADS" \
    -e MKL_DYNAMIC=$MKL_DYNAMIC \
    -v "$PWD":/data \
    rhee/deepdream python -u /data/deepdream.py --output=$b.output $input $iter $scale $model $guide

  sh run-cleanup.sh "$input"

done
