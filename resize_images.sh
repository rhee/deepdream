:
convert input_orig.jpg -colorspace RGB -resize 1024x999\> -colorspace sRGB input.jpg
convert guide_orig.jpg -colorspace RGB -gravity south -extent 2500x2500 -resize 320x320\> guide.jpg
