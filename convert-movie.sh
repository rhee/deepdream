#!/bin/sh

input_mp4="$1"; shift
output_mp4="$1"; shift

setopt shwordsplit 2>/dev/null
shopt -s nullglob 2>/dev/null

_ffmpeg=ffmpeg
avconv --help >/dev/null 2>&1 && ffmpeg=avconv

_ffmpeg_threads="-threads 8"
_ffmpeg_quality="-i_qfactor 0.71 -qcomp 0.6 -qmin 10 -qmax 63 -qdiff 4 -trellis 0"
_ffmpeg_output="-vcodec libx264 -pix_fmt yuv420p -profile:v baseline -b:v 1000k"
_ffmpeg_twopass="-movflags faststart"

exec $_ffmpeg $_ffmpeg_threads -i "$input_mp4" $_ffmpeg_input $_ffmpeg_quality $_ffmpeg_output "$output_mp4" $ffmpeg_twopass </dev/null
