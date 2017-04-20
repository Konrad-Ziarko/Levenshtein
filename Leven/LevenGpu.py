from numba import cuda
from numba import *
import numpy as np
from timeit import default_timer as timer
import math

#cuda.jit('uint32(uint32[:], uint32[:],uint32, uint32, uint32[:,:])', device=True)
def leven(A, B, w1, w2, metric):#Levenshtein algorithm len(A) = len(B)
    #initial data
    for i in range(0, w1):
        metric[i][0] = i
        metric[0][i] = i

    #computing metric value - Leven.Alg.
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
    #pattern len
    wordLen = len(word)
    #how many thread can work in parallel
    maxPos = len(line)-wordLen+1
    #??
    i = cuda.gridDim.x * cuda.blockDim.x + cuda.threadIdx.x;
    #call gpu function
    metric_values[i] = leven_gpu(word, line[i:i+wordLen], wordLen+1, wordLen, metric[:,:,i])#throws error
        


def main():

    print(cuda.detect())#is_available()
    #search patern
    pattern = "123456"
    #data for pattern search
    data_stream = "abcdef"
    #making more data, to stress cpu/gpu
    data_stream = data_stream*12500 #*20000
   
    pattern_array = []
    data_stream_array = []

    #string to unicode array
    for i in pattern:
        pattern_array.append(ord(i))
    for i in data_stream:
        data_stream_array.append(ord(i))

    #python array to np.array
    pattern_array = np.array(pattern_array, dtype=np.uint32)
    data_stream_array = np.array(data_stream_array, dtype=np.uint32)
    
    #array for metric values returned by gpu
    metric_values = np.zeros((len(data_stream_array)-len(pattern_array)+1), dtype = np.uint32)

    #how many thread can work in parallel, also 3dim of M matrix (code below)
    maxPos = len(data_stream)-len(pattern)+1
    #??
    tmp = cuda.get_current_device()
    #print(cuda.list_devices())
    
    #blockdim = (32,32)
    #griddim = (maxPos/(32*32*16)+1,16)

    threads_per_block = 32
    blocks_per_grid = (maxPos + (threads_per_block - 1)) # threadperblock

    #Levenshtein matrix for each thread => 3dim matrix => 2dim [(len(patern_array) x len(patern_array))] for each thread * num_of_threads
    M = np.zeros((len(pattern_array), len(pattern_array), len(data_stream_array)-len(pattern_array)+1), dtype = np.uint32)

    start = timer()

    d_stream = cuda.stream()

    #sending arrays to gpu
    d_metric_values = cuda.to_device(metric_values, stream = d_stream)
    d_M = cuda.to_device(M)
    d_array1 = cuda.to_device(pattern_array)
    d_array2 = cuda.to_device(data_stream_array)


    #call kernel
    leven_kernel[blocks_per_grid, threads_per_block](d_array1, d_array2, d_metric_values, d_M)

    #d_array1.copy_to_device()
    #d_array2.copy_to_device()
    d_metric_values.copy_to_host(metric_values, stream = d_stream)
    #d_M.to_host()
    d_stream.synchronize()
    dt = timer() - start
    print ('\n', dt)
    #print (d_values, '\n', dt)
    
    print (min(metric_values))

if __name__ == '__main__':
    main()