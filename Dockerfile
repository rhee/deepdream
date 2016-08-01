FROM ubuntu:14.04
MAINTAINER Sang-Hoon RHEE <shr386+github@hotmail.com>
# Based on deepdream-docker by Herval Freire <hervalfreire@gmail.com>

# General dependencies, lots of them
RUN apt-get update --fix-missing -y -q && apt-get install -y -q git wget bzip2 ca-certificates
RUN apt-get install -y -q libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev libatlas-dev libzmq3-dev libboost-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler bc libopenblas-dev


# # Python + pip
# RUN apt-get install -y python python-dev python-pip python-numpy python-scipy

# Intel mkl
RUN mkdir -p /tmp/installer /opt/intel/licenses
COPY intel_mkl.lic /opt/intel/licenses
ADD l_mkl_11.3.3.210.tgz /tmp/install
RUN sed -i 's/^ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' /tmp/install/l_mkl_11.3.3.210/silent.cfg
RUN cd /tmp/install/l_mkl_11.3.3.210; ./install.sh --silent ./silent.cfg
RUN rm -fr /tmp/install

# NOTE: fixed mkl paths as hardcoded
ENV CPATH=/opt/intel/compilers_and_libraries_2016.3.210/linux/mkl/include
ENV LIBRARY_PATH=/opt/intel/compilers_and_libraries_2016.3.210/linux/tbb/lib/intel64/gcc4.4:/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64:/opt/intel/compilers_and_libraries_2016.3.210/linux/mkl/lib/intel64:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64:/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64_lin:/opt/intel/compilers_and_libraries_2016.3.210/linux/tbb/lib/intel64/gcc4.4:/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64:/opt/intel/compilers_and_libraries_2016.3.210/linux/mkl/lib/intel64:$LD_LIBRARY_PATH
ENV INTEL_LICENSE_FILE=/opt/intel/compilers_and_libraries_2016.3.210/linux/licenses:/opt/intel/licenses:/root/intel/licenses
ENV PATH=/opt/intel/compilers_and_libraries_2016.3.210/linux/bin/intel64:$PATH
ENV MKLROOT=/opt/intel/compilers_and_libraries_2016.3.210/linux/mkl
ENV NLSPATH=/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64/locale/%l_%t/%N:/opt/intel/compilers_and_libraries_2016.3.210/linux/mkl/lib/intel64/locale/%l_%t/%N:$NLSPATH

# Anaconda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

# path for rest 
ENV PATH /opt/conda/bin:$PATH

#RUN /opt/conda/bin/conda install --offline numpy scipy

# Caffe
RUN git clone https://github.com/BVLC/caffe.git /caffe
WORKDIR /caffe

# Caffe's Python dependencies...
RUN easy_install --upgrade pip
RUN pip install -r python/requirements.txt

# Build Caffe
RUN cp Makefile.config.example Makefile.config

# # Enable CPU-only + openblas (faster than atlas)
# RUN sed -i 's/# CPU_ONLY/CPU_ONLY/g' Makefile.config
# RUN sed -i 's/BLAS := atlas/BLAS := open/g' Makefile.config

# Enable CPU-only + mkl ( faster, better multicore utilization )
# Use anaconda in /opt/conda
#RUN sed -i 's/# CPU_ONLY/CPU_ONLY/g' Makefile.config
#RUN sed -i 's/BLAS := atlas/BLAS := mkl/g' Makefile.config
COPY Makefile.config.override Makefile.config.override
RUN echo 'include Makefile.config.override' >> Makefile.config

# finally, compile Caffe, pyCaffe
RUN make all; make pycaffe

# update, cleanup
RUN conda install notebook ipython-notebook
RUN apt-get -y -q clean

ENV PYTHONPATH=/caffe/python

ENV MKL_NUM_THREADS=8
ENV MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=8"
ENV MKL_DYNAMIC=FALSE

# Download model
RUN scripts/download_model_binary.py models/bvlc_googlenet

# more extra utilities
RUN apt-get -y -q install imagemagick
RUN apt-get -y -q clean

# install ffmpeg static rather than libav-tools
ADD http://johnvansickle.com/ffmpeg/builds/ffmpeg-git-64bit-static.tar.xz /tmp
RUN tar xJ -v -f /tmp/ffmpeg-git-64bit-static.tar.xz -C /opt && rm -fv /tmp/ffmpeg-git-64bit-static.tar.xz && ln -s /opt/ffmpeg-git-20160731-64bit-static/ffmpeg /usr/local/bin

# install node-lts
ADD https://nodejs.org/dist/v4.4.7/node-v4.4.7-linux-x64.tar.xz /tmp
RUN tar xJ -v -f /tmp/node-v4.4.7-linux-x64.tar.xz -C /opt && rm -fv /tmp/node-v4.4.7-linux-x64.tar.xz && mkdir -p /etc/profile.d && echo 'PATH=/opt/node-v4.4.7-linux-x64/bin:$PATH; export PATH' > /etc/profile.d/node-lts.sh
ENV PATH=/opt/node-v4.4.7-linux-x64/bin:$PATH

#EXPOSE 8888
VOLUME ["/data"]

WORKDIR /data
