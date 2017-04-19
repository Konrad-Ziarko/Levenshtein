from numba import cuda
from numba import *
import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer

cuda.jit('uint32(uint32[:], uint32[:],uint32,uint32)', device=True)
def leven(A, B, h, w, out_val):
    i = j = 0
    
    metric = [[0 for x in range(w)] for y in range(h)] 

    #start = timer()
    for i in range(0, len(A)+1):
        metric[i][0] = i
    for j in range(0, len(B)+1):
        metric[0][j] = j

    for i in range(1, len(A)+1):    
        for j in range(1, len(B)+1):
            if A[i-1] == B[j-1]:
                cost = 0
            else:
                cost = 1
            metric[i][j] = min(metric[i-1][j]+1, metric[i][j-1]+1, metric[i-1][j-1] + cost)

    #vectoradd_time = timer() - start
    #print("Time:%f" % vectoradd_time)
    #print(metric)
    return metric[len(A)][len(B)]   
leven_gpu = cuda.jit(restype=uint32, argtypes=[uint32[:], uint32[:]], device=True)(leven)


@cuda.jit(argtypes=[uint32[:], uint32[:], uint32[:]])
def leven_kernel(word, line, metric_values):
    wordLen = len(word)
    maxPos = len(line)-wordLen+1
    for i in range(maxPos):
        metric_values[i] = leven(word, line[i:i+wordLen])         
        


def main():
    string1 = "abcdefgh"
    string2 = "abcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcba"
    string2 = string2*20000


    list1 = []
    list2 = []
    for i in string1:
        list1.append(ord(i))
    for i in string2:
        list2.append(ord(i))

    A = np.array(list1, dtype=np.uint32)
    B = np.array(list2, dtype=np.uint32)
    h = A.shape[0]+1
    w = B.shape[0]+1
    values = np.zeros((len(B)-len(A)+1), dtype = np.uint32)

    blockdim = (len(string2)-len(string1)+1, 1)
    griddim = (32,16)

    start = timer()

    d_values = cuda.to_device(values)
    leven_kernel[griddim, blockdim](A, B,h,w, values)
    d_values.to_host()
    dt = timer() - start
    print ('\n', dt)
    #print (d_values, '\n', dt)


if __name__ == '__main__':
    main()