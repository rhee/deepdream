:

setopt shwordsplit 2>/dev/null
shopt -s nullglob 2>/dev/null

MKL_NUM_THREADS=8
MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=8"
MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS MKL_DOMAIN_NUM_THREADS MKL_DYNAMIC

# lower level than the defualt inception_4c/output,
# gets more patternish output
#model=--model=inception_3b/5x5_reduce

#model=--model=inception_4c/output
#model=--model=inception_4d/output
#model=--model=inception_5a/output

# guide image binds the result, not an interesting
#guide=--guide=guide.jpg

iter=2400 # 100sec = 1:40sec
scale=0.05

set -e
set -x

for input in "$@"; do

  b=$(basename $input .jpg)

  python -u /data/deepdream.py --output=$b.output $input $iter $scale $model $guide

  ./cleanup.sh "$input"

done
