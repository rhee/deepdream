#!/bin/sh

for input in "$@"; do

  b=$(basename $input .jpg)

  cp $b.output/tmp.prototxt $b.tmp.prototxt
  ffmpeg -f image2 -r 24 -i $b.output/%04d.jpg -vcodec libx264 $b.output.mp4
  sudo rm -fr $b.output

done

