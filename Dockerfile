FROM ubuntu:14.04
MAINTAINER Herval Freire <hervalfreire@gmail.com>

# General dependencies, lots of them
RUN apt-get update && apt-get install -y git
RUN apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev libatlas-dev libzmq3-dev libboost-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler bc libopenblas-dev


# Python + pip
RUN apt-get install -y python python-dev python-pip python-numpy python-scipy


# Intel mkl
RUN mkdir -p /tmp/installer /opt/intel/licenses
COPY intel_mkl.lic /opt/intel/licenses
ADD l_mkl_11.3.3.210.tgz /tmp/install
RUN cd /tmp/install/l_mkl_11.3.3.210; sed -i 's/^ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' ./silent.cfg; ./install.sh --silent ./silent.cfg

# Caffe
RUN git clone https://github.com/BVLC/caffe.git /caffe
WORKDIR /caffe
RUN cp Makefile.config.example Makefile.config
RUN easy_install --upgrade pip

# Enable CPU-only + openblas (faster than atlas)
RUN sed -i 's/# CPU_ONLY/CPU_ONLY/g' Makefile.config
#RUN sed -i 's/BLAS := atlas/BLAS := open/g' Makefile.config
RUN sed -i 's/BLAS := atlas/BLAS := mkl/g' Makefile.config

# Caffe's Python dependencies...
RUN pip install -r python/requirements.txt

# mkl path set
ENV COMPILERVARS_ARCHITECTURE=intel64
ENV COMPILERVARS_PLATFORM=linux
RUN . /opt/intel/bin/compilervars.sh; make all; make pycaffe

ENV PYTHONPATH=/caffe/python

# Download model
RUN scripts/download_model_binary.py models/bvlc_googlenet

COPY deepdream.py /
COPY deepdream.sh /

VOLUME ["/data"]

WORKDIR /data

CMD ["/deepdream.sh", "input.jpg", "50", "0.05", "inception_4c/output"]
