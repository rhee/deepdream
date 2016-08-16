# Source: Google Deepdream code @ https://github.com/google/deepdream/
# Slightly modified in order to be run inside the container as a script instead of an IPython Notebook

from __future__ import print_function
import sys, os

import argparse
import traceback

import numpy as np
import PIL.Image

import deepdream
import catalogue

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='dump all patterns by layers')
    parser.add_argument('input_file', type=str, default='random.jpg')
    parser.add_argument('output_dir', type=str, default='layers')
    parser.add_argument('--amplify', type=int, default=3)
    parser.add_argument('--model_dir', type=str, default='bvlc_googlenet')
    parser.add_argument('--net_basename', type=str, default='deploy.prototxt')
    parser.add_argument('--param_basename', type=str, default='bvlc_googlenet.caffemodel')

    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir
    amplify = args.amplify
    model_dir = args.model_dir
    net_basename = args.net_basename
    param_basename = args.param_basename

    try: os.makedirs(output_dir)
    except: pass

    net, _ = deepdream.make_net(model_dir, net_basename, param_basename)

    img = np.float32(PIL.Image.open(input_file))

    output_file = '000_orig.jpg'
    PIL.Image.fromarray(np.uint8(img)).save(os.path.join(output_dir,output_file))

    i = 1
    for layer in net.blobs.keys():
        frame = img.copy()
        try:
            print('layer:',layer)
            for amplify_i in xrange(amplify):
                frame = deepdream.deepdream(net, frame, end=layer, objective=objective_L2)
            output_file = '%03d_%s.jpg' % (i,layer.replace('/','_'),)
            PIL.Image.fromarray(np.uint8(frame)).save(os.path.join(output_dir,output_file))
            print('wrote:',output_file,i)
            i += 1
        except:
            traceback.print_exc()
        del frame

    catalogue.make_cataloge(output_dir)

# vim: set sw=4 sts=4 ts=8 et ft=python
