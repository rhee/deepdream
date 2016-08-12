# Source: Google Deepdream code @ https://github.com/google/deepdream/
# Slightly modified in order to be run inside the container as a script instead of an IPython Notebook

from __future__ import print_function
import sys, os

caffe_root = os.getenv('CAFFE_ROOT') # this file should be run from {caffe_root}/examples (otherwise change this line)
use_cuda = os.getenv('USE_CUDA')

import traceback
import argparse

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image

from google.protobuf import text_format
import time

import nperf

###

caffe_python_path = caffe_root + 'python'
if caffe_python_path not in sys.path:
    sys.path.insert(0, caffe_python_path)

import caffe

###

if use_cuda:
    sys.stderr.write('USE_CUDA' + '\n')
    # try enable GPU
    try:
        GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
        caffe.set_mode_gpu()
        caffe.set_device(GPU_ID)
        use_cuda = True
    except:
        traceback.print_exc()

###

# default objective
def objective_L2(dst): dst.diff[:] = dst.data

def make_objective_guided(net, model_name, guide_image):

    h, w = guide_image.shape[:2]
    src, dst = net.blobs['data'], net.blobs[model_name]
    src.reshape(1,3,h,w)
    src.data[0] = preprocess(net, guide_image)
    net.forward(end=model_name)
    guide_features = dst.data[0].copy()

    def objective_guided(dst):
        x = dst.data[0].copy()
        y = guide_features
        ch = x.shape[0]
        x = x.reshape(ch,-1)
        y = y.reshape(ch,-1)
        A = x.T.dot(y) # compute the matrix of dot-products with guide features
        dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

    return objective_guided

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''
    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
    net.forward(end=end)
    objective(dst) # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)
        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail

        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base

    # returning the resulting image
    return deprocess(net, src.data[0])


def make_net():
    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".

    model_path = caffe_root + 'models/bvlc_googlenet/' # substitute your path here
    net_fn   = model_path + 'deploy.prototxt'
    param_fn = model_path + 'bvlc_googlenet.caffemodel'

    #!(cd $CAFFE_ROOT ; scripts/download_model_binary.py models/bvlc_googlenet)
    os.system('cd $CAFFE_ROOT; scripts/download_model_binary.py models/bvlc_googlenet')

    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True

    new_model_file = 'prototxt'
    open(new_model_file, 'w').write(str(model))

    net = caffe.Classifier(new_model_file, param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

    return net

####################

net = make_net()

input_file = 'input01.jpg'
output_dir = 'layer-by-layer'

try: os.makedirs(output_dir)
except: pass

img = np.float32(PIL.Image.open(input_file))

output_file = 'orig.jpg'
PIL.Image.fromarray(np.uint8(img)).save(os.path.join(output_dir,output_file))

for layer in net.blobs.keys():
    try:
        frame = img.copy()
        frame = deepdream(net, frame, end=layer, objective=objective_L2)
        frame = deepdream(net, frame, end=layer, objective=objective_L2)
        frame = deepdream(net, frame, end=layer, objective=objective_L2)
        output_file = layer.replace('/','_') + '.jpg'
        PIL.Image.fromarray(np.uint8(frame)).save(os.path.join(output_dir,output_file))
    except:
        traceback.print_exc()

# Emacs:
# Local Variables:
# mode: python
# c-basic-offset: 4
# End:
# vim: sw=4 sts=4 ts=8 et ft=python
