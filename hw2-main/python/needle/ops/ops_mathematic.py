"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        input, = node.inputs
        return out_grad * self.scalar * power_scalar(input, self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        lhs, rhs = node.inputs
        return divide(out_grad, rhs), out_grad*(-lhs) / power_scalar(rhs, 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        transpose_axes = [i for i in range(len(a.shape))]
        if self.axes:
            transpose_axes[self.axes[0]] = self.axes[1]
            transpose_axes[self.axes[1]] = self.axes[0]
        else:
            transpose_axes[-1], transpose_axes[-2] = transpose_axes[-2], transpose_axes[-1]
        return array_api.transpose(a, axes=transpose_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        input = node.inputs[0]
        return reshape(out_grad, input.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION   错哪里了----
        # raise NotImplementedError()
        # input_shape = node.inputs[0].shape
        # input_shape_len = len(input_shape) - 1
        # reduce_axes = []

        # for idx in range(len(out_grad.shape)-1, -1, -1):
        #     if input_shape_len < 0:
        #         reduce_axes.append(idx)
        #         continue

        #     if input_shape[input_shape_len] != out_grad.shape[idx]:
        #         reduce_axes.append(idx)
            
        #     input_shape_len -= 1

        # return reshape(summation(out_grad, reduce_axes), input_shape)
        ### END YOUR SOLUTION
        reduce_axes = []
        input, = node.inputs
        input_shape = input.shape
        out_shape = out_grad.shape
        s, n = 0, len(input_shape)
        for i, shape in enumerate(out_shape):
            if s == n:
                reduce_axes.append(i)
            else:
                if input_shape[s] != shape:
                    reduce_axes.append(i)
                s += 1

        return reshape(summation(out_grad, tuple(reduce_axes)), input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION  "test_nn_batchnorm" error
        # input_shape = node.inputs[0].shape
        # reshape_shape = list(input_shape)
        # if self.axes:
        #     if type(self.axes) == int:
        #         reshape_shape[self.axes] =1
        #     else:
        #         for i in list(self.axes):
        #             reshape_shape[i] = 1
        # else:
        #     reshape_shape = [1 for i in range(len(input_shape))]

        # out_grad = reshape(out_grad, reshape_shape)
        # return broadcast_to(out_grad, input_shape)
        ### END YOUR SOLUTION
        reshape_shape = []
        input, = node.inputs
        input_shape = input.shape
        out_shape = out_grad.shape
        s, n = 0, len(out_shape)
        for i, shape in enumerate(input_shape):
            if s < n:
                if shape == out_shape[s]:
                    reshape_shape.append(shape)
                    s += 1
                else:
                    reshape_shape.append(1)
            else:
                reshape_shape.append(1)

        return broadcast_to(reshape(out_grad, reshape_shape), input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        lhs, rhs = node.inputs
        lhs_lens = len(lhs.shape)
        rhs_lens = len(rhs.shape)
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)
        
        if lhs_lens > rhs_lens:
            sum_axes = tuple([i for i in range(lhs_lens - rhs_lens)])
            rhs_grad = summation(rhs_grad, axes=sum_axes)
            
        elif lhs_lens < rhs_lens:
            sum_axes = tuple([i for i in range(rhs_lens - lhs_lens)])
            lhs_grad = summation(lhs_grad, axes=sum_axes)
        
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        input = node.inputs[0]
        return divide(out_grad, input)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        input = node.inputs[0]
        return out_grad * exp(input)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        input = node.inputs[0]
        input = input.realize_cached_data()
        mask = Tensor(input > 0)
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
