#!/bin/sh

_ffmpeg=ffmpeg
avconv --help >/dev/null 2>&1 && _ffmpeg=avconv

for input in "$@"; do

  b=$(basename $input .jpg)

  cp $b.output/prototxt $b.prototxt
  $_ffmpeg -f image2 -r 24 -i $b.output/'%04d.jpg' -vcodec libx264 -pix_fmt yuv420p -profile:v baseline $b.output.mp4 </dev/null &&
    sudo rm -fr $b.output

done

