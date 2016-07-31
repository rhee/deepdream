#!/bin/sh

setopt shwordsplit 2>/dev/null
shopt -s nullglob 2>/dev/null

ffmpeg=ffmpeg
avconv --help >/dev/null 2>&1 && ffmpeg=avconv

for input in "$@"; do

  b=$(basename $input .jpg)

  cp $b.output/prototxt $b.prototxt
  $ffmpeg -f image2 -pattern_type glob -r 24 -i $b.output/'*.jpg' -vcodec libx264 $b.output.mp4
  sudo rm -fr $b.output

done

