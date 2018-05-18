#!/usr/bin/env sh

#./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt -step=one

#fine tune after pruned
#./build/tools/caffe train --solver=examples/mnist/lenet_fixed_solver.prototxt --weights=examples/mnist/lenet_iter_10000_fixed.caffemodel -step="two"

#test
./build/tools/caffe time -model=examples/alexnet/deploy.prototxt -weights=examples/alexnet/bvlc_alexnet.caffemodel -step="one" -gpu 0
