:
if [ -z "$2" ]; then
  echo "Usage: make-movie.sh dir mp4" 1>&2
  exit 1
fi

input_dir="$1"
output_mp4="$2"

exec ffmpeg -f image2 -pattern_type glob -r 24 -i "$input_dir"'/*.jpg' -vcodec libx264 "$output_mp4"
