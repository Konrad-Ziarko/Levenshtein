from numba import cuda
from numba import *
import numpy as np
from timeit import default_timer as timer

#cuda.jit('uint8(uint8[:], uint8[:],uint8, uint8, uint8[:,:])', device=True)
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
leven_gpu = cuda.jit(restype=uint8, argtypes=[uint8[:], uint8[:],uint8 ,uint8,uint8[:,:]], device=True)(leven)


@cuda.jit(argtypes=[uint8[:], uint8[:], uint8[:], uint8[:,:,:]])
def leven_kernel(word, line, metric_values, metric):
    #pattern len
    word_len = len(word)
    #how many thread can work in parallel
    maxPos = len(line)-word_len+1
    #??

    i = cuda.grid(1)
    #i = cuda.gridDim.x * cuda.blockDim.x + cuda.threadIdx.x;
    #call gpu function
    metric_values[i] = leven_gpu(word, line[i:i+word_len], word_len+1, word_len, metric[:,:,i])



def main():

    print(cuda.detect())#is_available()
    #search patern
    pattern = "1234"
    #data for pattern search
    data_stream = "abcdefghijk"
    #making more data, to stress cpu/gpu
    data_stream = data_stream*20000 #800 #*1000 #unknown_cuda_error
    pattern_array = []
    data_stream_array = []
    
    #string to unicode array
    for i in pattern:
        pattern_array.append(ord(i))
    for i in data_stream:
        data_stream_array.append(ord(i))

    print(pattern_array)
    #python array to np.array
    pattern_array = np.array(pattern_array, dtype=np.uint8)
    data_stream_array = np.array(data_stream_array, dtype=np.uint8)
    
    #array for metric values returned by gpu
    metric_values = np.zeros((len(data_stream_array)-len(pattern_array)+1), dtype = np.uint8)

    #how many thread can work in parallel, also 3dim of M matrix (code below)
    maxPos = len(data_stream)-len(pattern)+1
    #tmp = cuda.get_current_device()
    #print(cuda.list_devices())

    threads_per_block = 64
    blocks_per_grid = (maxPos + (threads_per_block - 1)) # threadperblock

    print(maxPos)
    print(maxPos-len(pattern_array))
    print(blocks_per_grid)

    #Levenshtein matrix for each thread => 3dim matrix => 2dim [(len(patern_array) x len(patern_array))] for each thread * num_of_threads
    M = np.zeros((len(pattern_array), len(pattern_array), len(data_stream_array)-len(pattern_array)+1), dtype = np.uint8)

    start = timer()


    #d_stream = cuda.stream()
    stream = cuda.stream()
    with stream.auto_synchronize():
        d_metric_values = cuda.to_device(metric_values,stream)
        d_M = cuda.to_device(M, stream) 
        d_array1 = cuda.to_device(pattern_array, stream)
        d_array2 = cuda.to_device(data_stream_array, stream)
        
        new = np.zeros((metric_values.shape[0],) ,dtype=np.uint8)
        leven_kernel[blocks_per_grid, threads_per_block, stream](d_array1, d_array2, d_metric_values, d_M)
        d_metric_values.copy_to_host(new, stream=stream)

    dt = timer() - start
    print ('\n', dt)
    #print (d_values, '\n', dt)
    
    print (min(metric_values))

if __name__ == '__main__':
    main()