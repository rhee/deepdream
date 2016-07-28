# Google Deepdream + Docker

A Docker Container to run Google's [Deepdream](https://github.com/google/deepdream/). This avoids having to setup all the dependencies (including GPU drivers, Python, Caffe, etc) in your OS of choice, so you can skip right to the fun part!


## Installing

The only dependency you need is [Docker](https://www.docker.com/).


## Building locally

```
docker build -t rhee/deepdream .
```


## Running

```
docker run \
    --name rhee/deepdream \
    -e OPENBLAS_NUM_THREADS=8 \
    -v "$PWD":/data rhee/deepdream python -u /deepdream.py input.jpg 20 0.10 inception_3b/5x5_reduce
```

*Note*: Depending on how much memory your machine has, you might run into problems with high-res images. In my case, processing failed for a 12mp image. Either stick to smaller images or buy more RAM ;-)


The output of the script will be written to the `output/` subfolder. Enjoy!
