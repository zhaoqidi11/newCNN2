import sys
import numpy as np
sys.path.insert(0, '/home/C3D/C3D-v1.1/models/python')
import caffe
if __name__ == '__main__':


    model_def = 'dsbd_test_resnet18.prototxt'
    model_weights = 'from_scratch11_c3d_sbd_iter_14000.caffemodel'

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))

    transformer.set_mean('data', np.array([73, 84, 91]))

    transformer.set_raw_scale('data', 255)

    transformer.set_channel_swap('data', (2, 1, 0))


    print 'a'