:
if [ -z "$2" ]; then
  echo "Usage: make-movie.sh pattern output-mp4 [framerate]" 1>&2
  exit 1
fi

pattern="$1"
output_mp4="$2"
frame_rate="$3"

test -z "$frame_rate" && frame_rate=2

###

setopt shwordsplit 2>/dev/null
shopt -s nullglob 2>/dev/null

_ffmpeg=ffmpeg
avconv --help >/dev/null 2>&1 && ffmpeg=avconv

###

_ffmpeg_threads="-threads 8"
#_ffmpeg_input="-f image2 -r ${frame_rate} -i"
_ffmpeg_input="-f image2 -r ${frame_rate} -pattern_type glob -i"
#_ffmpeg_quality="-i_qfactor 0.71 -qcomp 0.6 -qmin 10 -qmax 63 -qdiff 4 -trellis 0"
_ffmpeg_output="-vcodec libx264 -pix_fmt yuv420p -profile:v baseline" # "-b:v 1000k"
_ffmpeg_twopass="-movflags faststart"

set -x
exec $_ffmpeg $_ffmpeg_threads $_ffmpeg_input "$pattern" $_ffmpeg_quality $_ffmpeg_output "$output_mp4" $ffmpeg_twopass </dev/null
