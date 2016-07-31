:
if [ -z "$2" ]; then
  echo "Usage: resize-image.sh input output [geometry]" 1>&2
  exit 1
fi
input="$1"
output="$2"
geometry="$3"
test -z "$geometry" && geometry=1024x576
convert "$input" -colorspace RGB -resize 2048x1152\> -resize $geometry\> -gravity center -extent $geometry -colorspace sRGB "$output"
