"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue
            if param not in self.u:
                self.u[param] = ndl.zeros_like(param.grad)

            param_grad = param.grad.detach().cached_data + self.weight_decay * param.detach().cached_data
            self.u[param] = ndl.Tensor(self.momentum * self.u[param] + (1 - self.momentum) * param_grad, dtype='float32')
            param.data -= self.lr * self.u[param]

        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        '''
        L2正则化, 也叫做权重衰减。
        '''
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        for param in self.params:
            total_norm = 0.5 * np.sqrt(((param.detach().cached_data) * (param.detach().cached_data)).sum())
            clip_coef = max_norm / total_norm # 裁剪系数
            param = clip_coef * param
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()

        ### END YOUR SOLUTION