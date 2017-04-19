from numba import cuda
from numba import *
import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer

from Leven import *
#from LevenJit import *
#from LevenGpu import *

def main():
    string1 = "abcdefgh"
    string2 = "abcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcba"
    string2 = string2*20


    list1 = []
    list2 = []
    for i in string1:
        list1.append(ord(i))
    for i in string2:
        list2.append(ord(i))

    #A = np.array(list1, dtype=np.uint32)
    #B = np.array(list2, dtype=np.uint32)
    A = list1
    B = list2

    start = timer()
    metric = leven_dist(A,B)
    dt = timer() - start

    print (metric, '\n', dt)

    #jit
    """
    values = np.zeros((len(B)-len(A)+1), dtype = np.uint32)

    start = timer()
    leven_jit(A, B, values)
    dt = timer() - start

    print (values, '\n', dt)
    """
    #gpu

    values = np.zeros((len(B)-len(A)+1), dtype = np.uint32)

    #blockdim = (len(string2)-len(string1)+1, 1)
    #griddim = (32,16)

    start = timer()

    d_values = cuda.to_device(values)
    #leven_kernel[griddim, blockdim](A, B, values)
    d_values.to_host()
    dt = timer() - start
    print ('\n', dt)
    #print (d_values, '\n', dt)


if __name__ == '__main__':
    main()