#!/bin/sh

if [ -z "$3" ]; then
  echo "Usage: make-movie.sh input-dir output-mp4 framerate" 1>&2
  exit 1
fi

input_dir="$1"; shift
output_mp4="$1"; shift
frame_rate="$1"; shift

###

setopt shwordsplit 2>/dev/null
shopt -s nullglob 2>/dev/null

_ffmpeg=ffmpeg

###

_ffmpeg_threads="-threads 8"
_ffmpeg_input="-f image2 -r ${frame_rate} -pattern_type glob -i"
# _ffmpeg_quality="-i_qfactor 0.71 -qcomp 0.6 -qmin 10 -qmax 63 -qdiff 4 -trellis 0"
_ffmpeg_output="-vcodec libx264 -pix_fmt yuv420p -profile:v baseline" # "-b:v 1000k"
_ffmpeg_twopass="-movflags faststart"

set -x
exec $_ffmpeg $_ffmpeg_threads $_ffmpeg_input "$input_dir"/'*.jpg' $_ffmpeg_quality $_ffmpeg_output "$output_mp4" $ffmpeg_twopass "$@" </dev/null
