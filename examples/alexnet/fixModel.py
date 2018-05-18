import types
import collections
def flatten(x):
    def iselement(e):
        return not(isinstance(e, collections.Iterable) and not isinstance(e, str))
    for el in x:
        if iselement(el):
            yield el
        else:
            yield from flatten(el)
#def flatten(x):
#    result = []
#    for el in x:
#        if isinstance(x, collections.Iterable) and not isinstance(el, str):
#            result.extend(flatten(el))
#        else:
#            result.append(el)
#    return result
#from compiler.ast import flatten
import numpy as np
import matplotlib.pyplot as plt
caffe_root='../../'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
#model_def=caffe_root+'test_alexnet/bvlc_alexnet/deploy.prototxt'
#model_caffe=caffe_root+'test_alexnet/bvlc_alexnet/alexnet_train_iter_4000.caffemodel'

#model_def=caffe_root+'examples/mnist/deploy.prototxt'
#model_caffe=caffe_root+'examples/mnist/lenet_iter_10000.caffemodel'

model_def=caffe_root+'models/bvlc_alexnet/deploy.prototxt'
model_caffe=caffe_root+'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
net=caffe.Net(model_def,model_caffe,caffe.TEST)

def getVpt(v,k):
    v=abs(v)
    vList=v.tolist()
    vList=flatten(vList)
    #vList.sort()
    vList = sorted(vList)
    k=(int)(k*len(vList))
    return vList[k]



for k,v in net.params.items():
    idx=v[0].data.shape
    print (k)
    print (type(v[0].data))
    cnt=0
    count=0
    vpt=getVpt(v[0].data,0.8)
    if len(idx)==3:
        for n_idx in range(0,idx[0]):
            for c_idx in range(0,idx[1]):
                for h_idx in range(0,idx[2]):
                    for w_idx in range(0,idx[3]):
                        count=count+1
                        if v[0].data[n_idx][c_idx][h_idx][w_idx]<vpt  and v[0].data[n_idx][c_idx][h_idx][w_idx]>-1*vpt:
                            cnt=cnt+1
                            v[0].data[n_idx][c_idx][h_idx][w_idx]=0.0
                            v[0].mask[n_idx][c_idx][h_idx][w_idx]=0.0
        print (cnt)
        print (count)
    elif len(idx)==2:
        for h_idx in range(0,idx[0]):
            for w_idx in range(0,idx[1]):
                count=count+1
                if v[0].data[h_idx][w_idx]< vpt and v[0].data[h_idx][w_idx]>-1*vpt:
                    cnt=cnt+1
                    v[0].data[h_idx][w_idx]=0.0
                    v[0].mask[h_idx][w_idx]=0.0
       # print (cnt)
       # print (count)
   # print (v[0].data)
net.save('fixed.caffemodel')


