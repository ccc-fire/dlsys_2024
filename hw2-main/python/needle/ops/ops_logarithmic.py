from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = array_api.max(Z, axis=self.axes, keepdims=1)
        ret = array_api.log(
            array_api.exp(Z - maxz).sum(axis=self.axes, keepdims=1)) + maxz
        # NOTE do not use squeeze, because it may remove the 1-size dimension in Z.shape
        if self.axes:   
            out_shape = [size for i, size in enumerate(Z.shape) if i not in self.axes]
        else:
            out_shape = ()
        ret.resize(tuple(out_shape))
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        if self.axes:
            shape = [1] * len(Z.shape)
            s = set(self.axes)
            j = 0
            for i in range(len(shape)):
                if i not in s:
                    shape[i] = node.shape[j]
                    j += 1
            node_new = node.reshape(shape)
            grad_new = out_grad.reshape(shape)
        else:
            node_new = node
            grad_new = out_grad
        # print(node.shape, Z.shape, node_new.shape, out_grad.shape)
        return grad_new * exp(Z - node_new)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

