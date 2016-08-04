:

iter=2400 # 100sec = 1:40sec
scale=0.05

set -e
set -x

for input in "$@"; do

  b=$(basename $input .jpg)

  python -u deepdream.py --output=$b.output $input $iter $scale $model $guide

  ./cleanup.sh "$input"

done
