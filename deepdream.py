# Source: Google Deepdream code @ https://github.com/google/deepdream/
# Slightly modified in order to be run inside the container as a script instead of an IPython Notebook

import sys, os
import argparse
from nperf.nperf import nperf

parser = argparse.ArgumentParser(description='deepdream demo')
parser.add_argument('--output', type=str, default='output', help='Output directory')
parser.add_argument('--model', type=str, default='auto', help='Model network name')
parser.add_argument('--guide', type=str, default='', help='Guide image')
parser.add_argument('input_file', default='input.png')
parser.add_argument('iterations', default=50)
parser.add_argument('scale', default=0.05)

args = parser.parse_args()

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image

from google.protobuf import text_format
import time

import caffe


# try enable GPU
try:
    GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
except:
    pass


check = nperf.nperf(interval = 60.0)

###

caffe_home = os.getenv('CAFFE_HOME')
model_path = caffe_home + '/models/bvlc_googlenet/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

models_nice = [
    'inception_4c/output',
    'inception_4d/output',
    'inception_5a/output'
]

models_choice = np.random.randint(0,len(models_nice))

###

output_dir = args.output
guide = args.guide
model_name = args.model
input_file = args.input_file
iterations = args.iterations
scale = args.scale

###

# make /data/output

try: os.makedirs(output_dir)
except: pass

print("Processing file: " + input_file)
print("Iterations = %s" % iterations)
print("Scale = %s" % scale)
print("Model = %s" % model_name)

img = np.float32(PIL.Image.open(input_file))

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True

open('%s/prototxt' % (output_dir,), 'w').write(str(model))

net = caffe.Classifier('%s/prototxt' % (output_dir,), param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# verify model name provided
if 'auto' != model_name:

    if not model_name in net.blobs.keys():
        sys.stderr.write('Invalid model name: %s' % (model_name,) + '\n')
        sys.stderr.write('Valid models are:' + repr(net.blobs.keys()) + '\n')
        sys.exit(-1)

if 'auto' != model_name and guide:

    guide_image = np.float32(PIL.Image.open(guide))
    h, w = guide_image.shape[:2]
    src, dst = net.blobs['data'], net.blobs[model_name]
    src.reshape(1,3,h,w)
    src.data[0] = preprocess(net, guide_image)
    net.forward(end=model_name)

    # global
    guide_features = dst.data[0].copy()

    def objective_guide(dst):
        x = dst.data[0].copy()
        y = guide_features
        ch = x.shape[0]
        x = x.reshape(ch,-1)
        y = y.reshape(ch,-1)
        A = x.T.dot(y) # compute the matrix of dot-products with guide features
        dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

    objective = objective_guide

else:

    def objective_L2(dst):
        dst.diff[:] = dst.data 

    objective = objective_L2

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True, objective=objective):
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

            # visualization
            #vis = deprocess(net, src.data[0])
            #if not clip: # adjust image contrast if clipping is disabled
            #    vis = vis*(255.0/np.percentile(vis, 99.98))
            #showarray(vis)
            #clear_output(wait=True)

            #print(octave, i, end #, vis.shape)

    def print_out(count, tlap):
        print 'snapshot:', octave, i, end

        check('deepdream', print_out)

        #print(octave, '*', end) #, vis.shape

        # extract details produced on the current octave
        detail = src.data[0]-octave_base

    # returning the resulting image
    return deprocess(net, src.data[0])

frame_i = 1

frame = img
PIL.Image.fromarray(np.uint8(frame)).save("%s/%04d.jpg"%(output_dir, frame_i))
frame_i += 1

h, w = frame.shape[:2]
s = float(scale) # scale coefficient

recovery_mode = False

for i in xrange(int(iterations)):
    #print "Step %d of %d is starting..." % (i, int(iterations))

    step_output_file = "%s/%04d.jpg"%(output_dir, frame_i)

    if os.path.exists(step_output_file):
        if not recovery_mode:
            recovery_mode = True
            sys.stderr.write('Found previous output. Assume recovery mode.' + '\n')
        frame_i += 1
        continue

    if recovery_mode and not os.path.exists(step_output_file):
        last_output_file = "%s/%04d.jpg"%(output_dir, frame_i - 1)
        frame = np.float32(PIL.Image.open(last_output_file))
        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
        recovery_mode = False
        sys.stderr.write('recovery_mode: continue from ' + step_output_file + '\n')

    if 'auto' == model_name:
        if np.random.randint(0, 120) == 0:
            models_choice = np.random.randint(0,len(models_nice))
            end = models_nice[models_choice]
        else:
            end = model_name

    frame = deepdream(net, frame, end=end, objective=objective)
    PIL.Image.fromarray(np.uint8(frame)).save(step_output_file)
    frame_i += 1

    frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)

    print("Step %d of %d is complete." % (i, int(iterations)))

#print "All done! Check the " + output_dir + " folder for results"

# Emacs:
# Local Variables:
# mode: python
# c-basic-offset: 4
# End:
# vim: sw=4 sts=4 ts=8 et ft=python
