# Source: Google Deepdream code @ https://github.com/google/deepdream/
# Slightly modified in order to be run inside the container as a script instead of an IPython Notebook

from __future__ import print_function
import sys, os

import argparse
from traceback import print_exc

import numpy as np
import PIL.Image

from deepdream import make_net, deepdream, objective_L2
from catalogue import make_catalogue

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='dump all patterns by layers')
    parser.add_argument('input_file', type=str, default='random.jpg')
    parser.add_argument('output_dir', type=str, default='layers')
    parser.add_argument('--amplify', type=int, default=3)
    parser.add_argument('--model_dir', type=str, default='bvlc_googlenet')
    parser.add_argument('--prototxt', type=str, default='deploy.prototxt')
    parser.add_argument('--caffemodel', type=str, default='*.caffemodel')

    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir
    amplify = args.amplify
    model_dir = args.model_dir
    prototxt = args.prototxt
    caffemodel = args.caffemodel

    try: os.makedirs(output_dir)
    except: pass

    net, _ = make_net(model_dir, prototxt, caffemodel)

    img = np.float32(PIL.Image.open(input_file))

    output_file = '000_orig.jpg'
    PIL.Image.fromarray(np.uint8(img)).save(os.path.join(output_dir,output_file))

    # InnerProduct layer crashes caffe, aborts
    blacklist_layers = [
        'fc6',
        'fc7',
        'fc8',
        'fc8_flickr',
        'fc9',
        'pool5/7x7_s1',
        'prob',
        'loss3/classifier_ftune',
        'fc-rcnn',
        'loss3/classifier',
        'ip3',
        'ip4',
        'ip5',
    ]

    i = 1
    for layer in net.blobs.keys():
        if layer in net._layer_names:
            try:
                output_file = '%03d_%s.jpg' % (i, layer.replace('/', '_'),)
                # skip if already created in prev session
                if os.path.exists(os.path.join(output_dir,output_file)):
                    i += 1
                    continue
                if layer in blacklist_layers:
                    print('skip:', layer)
                    continue
                print('layer:', layer, output_file)
                frame = img.copy()
                for amplify_i in xrange(amplify):
                    frame = deepdream(net, frame, end=layer, objective=objective_L2)
                PIL.Image.fromarray(np.uint8(frame)).save(os.path.join(output_dir,output_file))
                i += 1
            except KeyboardInterrupt:
                sys.exit(1)
            except:
                print_exc()

    make_catalogue(output_dir)

# vim: set sw=4 sts=4 ts=8 et ft=python
