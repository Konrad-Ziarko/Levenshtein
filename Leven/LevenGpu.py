from numba import cuda
from numba import *
import numpy as np
from timeit import default_timer as timer

#cuda.jit('uint32(uint32[:], uint32[:],uint32, uint32, uint32[:,:])', device=True)
def leven(A, B, w1, w2, metric):
    for i in range(0, w1):
        metric[i][0] = i
        metric[0][i] = i

    for i in range(1, w1):    
        for j in range(1, w1):
            if A[i-1] == B[j-1]:
                cost = 0
            else:
                cost = 1
            metric[i][j] = min(metric[i-1][j]+1, metric[i][j-1]+1, metric[i-1][j-1] + cost)
    return metric[w2][w2]
leven_gpu = cuda.jit(restype=uint32, argtypes=[uint32[:], uint32[:],uint32 ,uint32,uint32[:,:]], device=True)(leven)


@cuda.jit(argtypes=[uint32[:], uint32[:], uint32[:], uint32[:,:,:]])
def leven_kernel(word, line, metric_values, metric):
    wordLen = len(word)
    maxPos = len(line)-wordLen+1
    for i in range(maxPos):
        metric_values[i] = leven_gpu(word, line[i:i+wordLen], wordLen+1, wordLen)         
        


def main():
    string1 = "abcdefgh"
    string2 = "abcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcba"
    string2 = string2*20

    string1_array = []
    string2_array = []
    for i in string1:
        string1_array.append(ord(i))
    for i in string2:
        string2_array.append(ord(i))

    string1_array = np.array(string1_array, dtype=np.uint32)
    string2_array = np.array(string2_array, dtype=np.uint32)

    values = np.zeros((len(string2_array)-len(string1_array)+1), dtype = np.uint32)

    blockdim = (len(string2)-len(string1)+1, 1)
    griddim = (32,16)

    M = np.zeros((w1, w1, len(string2_array)-len(string1_array)+1), dtype = np.uint32)

    start = timer()

    d_values = cuda.to_device(values)
    d_M = cuda.to_device(M)
    d_array1 = cuda.to_device(string1_array)
    d_array2 = cuda.to_device(string2_array)

    leven_kernel[griddim, blockdim](d_array1, d_array2, d_values, d_M)

    d_array1.copy_to_device()
    d_array2.copy_to_device()
    d_values.to_host()
    d_M.to_host()

    dt = timer() - start
    print ('\n', dt)
    #print (d_values, '\n', dt)


if __name__ == '__main__':
    main()