#!/bin/sh

dir=$1

i=0
while test "$i" -lt 100
do
  ii=$(printf "%04d" $i)
  iii=$(printf "%04d00" $i)
  if ls "$dir/$ii"*.jpg >/dev/null 2>&1; then
    echo "$dir/$iii" 1>&2
    mkdir "$dir/$iii"
    mv "$dir/$ii"*.jpg "$dir/$iii"
  fi
  i=$(expr $i + 1)
done
