:
if [ -z "$2" ]; then
  echo "Usage: resize-image.sh input output [geometry]" 1>&2
  exit 1
fi
input="$1"
output="$2"
geometry="$3"
test -z "$geometry" && geometry=640x360
# enlarge to max (3840x2160), shrink to cover geometry, center crop by geometry
convert "$input" -colorspace RGB -resize 3840x2160\< -resize $geometry^ -gravity center -extent $geometry -colorspace sRGB "$output"
