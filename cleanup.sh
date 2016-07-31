#!/bin/sh

for input in "$@"; do

  b=$(basename $input .jpg)

  cp $b.output/tmp.prototxt $b.tmp.prototxt
  ffmpeg -f image2 -pattern_type glob -r 24 -i $b.output/'*.jpg' -vcodec libx264 $b.output.mp4
  sudo rm -fr $b.output

done

