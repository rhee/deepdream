#!/bin/sh

for input in "$@"; do

  b=$(basename $input .jpg)

  cp $b.output/prototxt $b.prototxt
  ./make-movie.sh "$b.output" "$b.output.mp4" &&
    rm -fr $b.output

done
