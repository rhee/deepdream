# Source: Google Deepdream code @ https://github.com/google/deepdream/
# Slightly modified in order to be run inside the container as a script instead of an IPython Notebook

import sys, os
import argparse

parser = argparse.ArgumentParser(description='deepdream demo')
parser.add_argument('--output', type=str, default='output', help='Output directory')
parser.add_argument('--guide', type=str, default='', help='Guide image')
parser.add_argument('--list-keys', help='list model names')
parser.add_argument('input', default='input.png')
parser.add_argument('iter', default=50)
parser.add_argument('scale', default=0.05)
parser.add_argument('model', default='inception_4c/output')

args = parser.parse_args()

#if len(sys.argv) < 5:
#    sys.stderr.write('''
#Usage: [python] deepdream.py input iter scale model
#Example: python deepdream.py input.jpg 50 0.05 inception_4c/output
#
#Type deepdream.py --list-keys to view the model names
#''')
#    sys.exit(1)
#
#if len(sys.argv) >= 2 and '--list-keys' == sys.argv[1]:
#    print(net.blobs.keys())
#    sys.exit(0)

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image

#from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import time

import caffe

output_dir = args.output
input_file = args.input #os.getenv('INPUT', 'input.png')
iterations = args.iter #os.getenv('ITER', 50)

try:
    iterations = int(iterations)
except ValueError:
    iterations = 50

scale = float(args.scale) #os.getenv('SCALE', 0.05)
try:
    scale = float(scale)
except ValueError:
    scale = 0.05

model_name = args.model #os.getenv('MODEL', 'inception_4c/output')
print "Processing file: " + input_file

guide = args.guide

img = np.float32(PIL.Image.open(input_file))

model_path = '/caffe/models/bvlc_googlenet/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

# make /data/output, /data/output/tmp
try: os.makedirs("%s/tmp" % (output_dir,))
except: pass

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('%s/tmp.prototxt' % (output_dir,), 'w').write(str(model))

net = caffe.Classifier('%s/tmp.prototxt' % (output_dir,), param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

def verifyModel(net, model):
    print "Verifying model: %s" % model
    keys = net.blobs.keys()
    if model in keys:
        print "Model %s is valid." %model
        return True
    else:
        print "Invalid model: %s.  Valid models are:" % model
        for k in keys:
            print k
        return False

if not verifyModel(net, model_name):
    sys.exit(0) #os._exit(1)

# now we can handle --list-keys
if args.list_keys:
    print(net.blobs.keys())
    sys.exit(0)

def showarray(a):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    millis = int(round(time.time() * 1000))
    filename = "%s/tmp/steps-%i.jpg" % (output_dir, millis)
    PIL.Image.fromarray(np.uint8(a)).save(filename)

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def objective_L2(dst):
    dst.diff[:] = dst.data 

def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''
    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
    net.forward(end=end)
    #dst.diff[:] = dst.data  # specify the optimization objective
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
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            showarray(vis)
            print octave, i, end, vis.shape
            #clear_output(wait=True)
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])


# guide
guide_features = None
guide_end = None

if guide:
	guide_end = 'inception_3b/output'
	guide_image = np.float32(PIL.Image.open(args.guide))
	#showarray(guide_image)
	h, w = guide_image.shape[:2]
	src, dst = net.blobs['data'], net.blobs[guide_end]
	src.reshape(1,3,h,w)
	src.data[0] = preprocess(net, guide_image)
	net.forward(end=guide_end)
	guide_features = dst.data[0].copy()

def objective_guide(dst):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

frame = img
frame_i = 0

h, w = frame.shape[:2]
s = float(scale) # scale coefficient
print "Entering dream mode..."
print "Iterations = %s" % iterations
print "Scale = %s" % scale
print "Model = %s" % model_name
for i in xrange(int(iterations)):
    print "Step %d of %d is starting..." % (i, int(iterations))
    if guide_features is None:
	frame = deepdream(net, frame) #, end=model_name)
    else:
	frame = deepdream(net, frame, end=guide_end, objective=objective_guide)
    PIL.Image.fromarray(np.uint8(frame)).save("%s/%04d.jpg"%(output_dir, frame_i))
    frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
    frame_i += 1
    print "Step %d of %d is complete." % (i, int(iterations))

print "All done! Check the " + output_dir + " folder for results"
