:
if [ -z "$2" ]; then
  echo "Usage: make-movie.sh input-dir output-mp4" 1>&2
  exit 1
fi

input_dir="$1"
output_mp4="$2"

###

setopt shwordsplit 2>/dev/null
shopt -s nullglob 2>/dev/null

ffmpeg=ffmpeg
avconv --help >/dev/null 2>&1 && ffmpeg=avconv

###

#exec $ffmpeg -f image2 -pattern_type glob -r 24 -i "$input_dir"'/*.jpg' -vcodec libx264 "$output_mp4"
exec $ffmpeg -f image2 -r 24 -i "$input_dir"'/%04d.jpg' -vcodec libx264 "$output_mp4" </dev/null
