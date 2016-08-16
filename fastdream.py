# Source: Google Deepdream code @ https://github.com/google/deepdream/
# Slightly modified in order to be run inside the container as a script instead of an IPython Notebook

from __future__ import print_function
import sys, os

import numpy as np
import PIL.Image
import scipy.ndimage as nd

from deepdream import make_net, objective_L2, make_objective_guided, deepdream
import nperf

def main(args):

    input_file = args.input_file
    output_dir = args.output_dir

    layer = args.layer
    iterations = args.iterations
    scale = args.scale
    guide = args.guide

    model_dir = args.model_dir
    prototxt = args.prototxt
    caffemodel = args.caffemodel

    try: os.makedirs(output_dir)
    except: pass

    print("Processing file: " + input_file)
    print("Iterations = %s" % iterations)
    print("Scale = %s" % scale)
    print("Model = %s" % layer)

    net, model = make_net(model_dir=model_dir, prototxt=prototxt, caffemodel=caffemodel)

    prototxt = os.path.join(output_dir, 'prototxt')
    open(prototxt, 'w').write(str(model))

    # verify model name provided
    if not layer in net.blobs.keys():
        sys.stderr.write('Invalid model name: %s' % (layer,) + '\n')
        sys.stderr.write('Valid models are:' + repr(net.blobs.keys()) + '\n')
        sys.exit(1)

    if guide:
        guide_image = np.float32(PIL.Image.open(guide))
        objective = make_objective_guided(net, layer, guide_image)
    else:
        objective = objective_L2

    ###

    def output_fn(i):
        return os.path.join(output_dir, '%04d.jpg' % (i,))

    frame = None
    i = 1
    while i < iterations + 1:
        if not os.path.exists(output_fn(i)):
            break
        i += 1
    
    if i > 1:
        frame = np.float32(PIL.Image.open(output_fn(i-1)))
        h, w = frame.shape[:2]
        frame = nd.affine_transform(frame, [1-scale,1-scale,1], [h*scale/2,w*scale/2,0], order=1)
    else:
        frame = np.float32(PIL.Image.open(input_file))
        h, w = frame.shape[:2]
        PIL.Image.fromarray(np.uint8(frame)).save(output_fn(1))

    # start next images
    check2 = nperf.nperf(interval = 30.0, maxcount = (iterations - i + 1))

    if os.getenv('USE_CUDA'):
        perf_tag2 = '[cuda] fastdream'
    else:
        perf_tag2 = '[cpu] fastdream'

    print('####################################')
    print('#   loop starts from', i)
    print('####################################')

    while i <= iterations:
        frame = deepdream(net, frame, end=layer, objective=objective)
        PIL.Image.fromarray(np.uint8(frame)).save(output_fn(i))
        # affine transform (zoom-in) before feed as next step input
        frame = nd.affine_transform(frame,[1 - scale, 1 - scale, 1],[h * scale / 2, w * scale / 2, 0],order=1)
        check2(perf_tag2)
        i += 1

    #print "All done! Check the " + output_dir + " folder for results"

if '__main__' == __name__:

    import argparse

    parser = argparse.ArgumentParser(description='deepdream demo')
    parser.add_argument('input_file', type=str, default='input.jpg')
    parser.add_argument('output_dir', type=str, default='output')
    parser.add_argument('--layer', type=str, default='inception_4c/output')
    parser.add_argument('--iterations', type=int, default=1440)
    parser.add_argument('--scale', type=float, default=0.05)
    parser.add_argument('--guide', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default='bvlc_googlenet')
    parser.add_argument('--prototxt', type=str, default='deploy.prototxt')
    parser.add_argument('--caffemodel', type=str, default='*.caffemodel')

    args = parser.parse_args()

    main(args)

# vim: sw=4 sts=4 ts=8 et ft=python
