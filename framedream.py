# Source: Google Deepdream code @ https://github.com/google/deepdream/
# Slightly modified in order to be run inside the container as a script instead of an IPython Notebook

from __future__ import print_function
import sys, os

import numpy as np
import PIL.Image
import scipy.ndimage as nd

import deepdream
import nperf

def main(args):

    input_dir = args.input_dir
    output_dir = args.output_dir
    layer = args.layer
    amplify = args.amplify

    model_dir = args.model_dir
    net_basename = args.net_basename
    param_basename = args.param_basename

    try: os.makedirs(output_dir)
    except: pass

    net, model = deepdream.make_net(model_dir=model_dir, net_basename=net_basename, param_basename=param_basename)
 
    # verify model name provided
    if not layer in net.blobs.keys():
        sys.stderr.write('Invalid model name: %s' % (layer,) + '\n')
        sys.stderr.write('Valid models are:' + repr(net.blobs.keys()) + '\n')
        sys.exit(1)

    ###

    files = [v for v in os.listdir(input_dir) if v.endswith('.jpg')]
    files.sort()

    # scan existing output images
    i = 0
    while i < len(files):
        f = files[i]
        output_file = os.path.join(output_dir, f)
        if not os.path.exists(output_file):
            break
        i += 1

    if i > 0:
        # make guide function from last image
        guide_image = np.float32(PIL.Image.open(os.path.join(input_dir,files[i-1])))
        objective = deepdream.make_objective_guided(net, layer, guide_image)
    else:
        # make initial L2 guide function
        objective = objective_L2

    # start next images
    check2 = nperf.nperf(interval = 30.0, maxcount = (len(files) - i) * amplify)

    if os.getenv('USE_CUDA'):
        perf_tag2 = '[cuda] framedream'
    else:
        perf_tag2 = '[cpu] framedream'

    print('####################################')
    print('#   loop starts from', i)
    print('####################################')

    while i < len(files):
        f = files[i]
        input_file = os.path.join(input_dir, f)
        output_file = os.path.join(output_dir, f)
        frame = np.float32(PIL.Image.open(input_file))
        print("processing:", f, frame.shape, layer, amplify)
        for short_i in xrange(amplify):
            frame = deepdream.deepdream(net, frame, end=layer, objective=objective)
            check2(perf_tag2)
        # use this frame as guide image for next iteration
        objective = deepdream.make_objective_guided(net, layer, frame)
        PIL.Image.fromarray(np.uint8(frame)).save(output_file)
        i += 1

if '__main__' == __name__:

    import argparse

    parser = argparse.ArgumentParser(description='deepdream demo')
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--layer', type=str, default='inception_4c/output', help='layer to reflect')
    parser.add_argument('--guide', type=str, default='', help='Guide image')
    parser.add_argument('--amplify', type=int, default=2)
    parser.add_argument('--model_dir', type=str, default='bvlc_googlenet')
    parser.add_argument('--net_basename', type=str, default='deploy.prototxt')
    parser.add_argument('--param_basename', type=str, default='bvlc_googlenet.caffemodel')

    args = parser.parse_args()

    main(args)

# vim: sw=4 sts=4 ts=8 et ft=python
