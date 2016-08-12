#!/bin/sh

for dir in "$@"; do

  cp $dir/prototxt $dir.prototxt
  ./make-movie.sh "$dir/*.jpg" "$dir.mp4" &&
    rm -fr "$dir"

done
