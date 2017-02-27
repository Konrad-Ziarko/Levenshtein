import numpy as np
from timeit import default_timer as timer
from numba import cuda, vectorize

@vectorize(['float32(float32, float32)'], target='cuda')
def VectorAdd(a,b):
    return a+b

def main():
    N = 30000000
    A=np.ones(N, dtype=np.float32)
    d_a = cuda.to_device(A);
    B=np.ones(N, dtype=np.float32)
    d_b = cuda.to_device(B);
    C=np.zeros(N, dtype=np.float32)
    d_c = cuda.device_array(A.shape[0])

    start = timer()
    d_c = VectorAdd(d_a,d_b);
    d_c.copy_to_host(C);
    vectoradd_time = timer() - start

    print("Time:%f" % vectoradd_time)

if __name__ == '__main__':
    main()