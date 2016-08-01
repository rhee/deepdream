# Google Deepdream + Docker

This project is forked from [herval/deepdream-docker](https://github.com/herval/deepdream-docker), and the description from the origin follows:

A Docker Container to run Google's [Deepdream](https://github.com/google/deepdream/). This avoids having to setup all the dependencies (including GPU drivers, Python, Caffe, etc) in your OS of choice, so you can skip right to the fun part!


## Installing

Requirements:

- [Docker](https://www.docker.com/).
- Intel(r) Math Kernel Library - You can register/download Non-comercial/Evaluation version from [Intel Math Kernel Library Page](https://software.intel.com/en-us/intel-mkl/)
    - Save the downloaded tgz file as `l_mkl_11.3.3.210.tgz` in build directory
    - Save the downloaded license file as `intel_mkl.lic` in build directory
    - You may want to change to downloaded tgz file name in `Dockerfile` if your downloaded file name not matches
- Added `anaconda` and `ipython-notebook` for interactive use and `ipynb` play.
    - `anaconda` has better `mkl` supports pre-installed
- Added `imagemagick` and `ffmpeg` for simple image conversions and inspections.

## Building locally

```
docker build -t rhee/deepdream .
```


## Running

Save image to start to project directory, and then run:

```
./run.sh <imagename>.jpg
```

see `run-docker.sh` to check options related to `docker`.

see `run.sh` to check `deepdream.py` options.

The output of the script will be written to the `<imagename>.output.mp4` file.

Comment on the memory requirements from [herval/deepdream-docker](https://github.com/herval/deepdream-docker):

    *Note*: Depending on how much memory your machine has, you might run into problems with high-res images. In my case, processing failed for a 12mp image. Either stick to smaller images or buy more RAM ;-)

