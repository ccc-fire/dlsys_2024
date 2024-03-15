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
        # raise NotImplementedError()

        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        log_sum_exp_z = array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=self.axes, keepdims=True)) + max_z
        log_sum_exp_z = array_api.squeeze(log_sum_exp_z, axis=self.axes)
        return log_sum_exp_z

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z, = node.inputs
        if self.axes:
            

            pass
        else:
            return out_grad * exp(Z - node)
        # ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

