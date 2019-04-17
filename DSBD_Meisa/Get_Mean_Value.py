import sys
sys.path.insert(0,'/media/user02/Volume/C3D/C3D-v1.0/python')
import caffe
import numpy as np
b = caffe.proto.caffe_pb2.BlobProto()
d = open('/media/user02/Volume/DSBD/models/mean.binaryproto','rb').read()
b.ParseFromString(d)
a = np.asarray(b.data)


print 'a'