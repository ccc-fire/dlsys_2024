import sys
sys.path.append('./python')
from needle import backend_ndarray as nd


x = nd.NDArray([1, 4, 3], device=nd.cpu())
x1 = nd.NDArray([0, 1, 2], device=nd.cpu())
print((x1 / x))

a = nd.NDArray.make((2, 3, 4), device = nd.cpu())
print(a)