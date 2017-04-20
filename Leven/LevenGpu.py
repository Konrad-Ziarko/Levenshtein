from numba import cuda
from numba import *
import numpy as np
from timeit import default_timer as timer

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
    #search patern
    pattern = "123456"
    #data for pattern search
    data_stream = "abcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcba"
    #making more data, to stress cpu/gpu
    data_stream = data_stream*20000 #*20000
   
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
    print(metric_values.shape[0])
    #how many thread can work in parallel, also 3dim of M matrix (code below)
    maxPos = len(data_stream)-len(pattern)+1
    #??
    blockdim = (maxPos,1)
    griddim = (1,1)

    #Levenshtein matrix for each thread => 3dim matrix => 2dim [(len(patern_array) x len(patern_array))] for each thread * num_of_threads
    M = np.zeros((len(pattern_array), len(pattern_array), len(data_stream_array)-len(pattern_array)+1), dtype = np.uint32)

    start = timer()

    #sending arrays to gpu
    d_values = cuda.to_device(metric_values)
    d_M = cuda.to_device(M)
    d_array1 = cuda.to_device(pattern_array)
    d_array2 = cuda.to_device(data_stream_array)

    #call kernel
    #leven_kernel[griddim, blockdim](d_array1, d_array2, d_values, d_M)

    #d_array1.copy_to_device()
    #d_array2.copy_to_device()
    d_values.to_host()
    #d_M.to_host()

    dt = timer() - start
    print ('\n', dt)
    #print (d_values, '\n', dt)


if __name__ == '__main__':
    main()