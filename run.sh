:

iter=1440 # 60sec

set -e
set -x

for input in "$@"; do

  b=$(basename $input .jpg)

  python -u deepdream.py $input $b.output $iter

  ./cleanup.sh "$b.output"

done
