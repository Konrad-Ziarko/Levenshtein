from numba import cuda
from numba import *
import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer


def leven(str1, str2):
    list1 = []
    list2 = []
    for i in str1:
        list1.append(ord(i))
    for i in str2:
        list2.append(ord(i))

    A = np.array(list1, dtype=np.int)
    B = np.array(list2, dtype=np.int)
    i = j = 0
    metric = np.zeros([A.shape[0]+1, B.shape[0]+1], dtype=np.int)

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

#leven_gpu = cuda.jit(restype=uint32, argtypes=[uint32, uint32], device=True)(leven)
#@cuda.jit(argtypes=[f8, f8, f8, f8, uint8[:,:], uint32])
@autojit
def leven_kernel(word, line, metric_values):
    
    wordLen = len(word)
    metric_value = wordLen+1
    maxPos = len(line)-wordLen+1


    for i in range(maxPos):
        metric_values[i] = leven(word, line[i:i+wordLen])
        
            
           




def mandel(x, y, max_iters):
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4:
      return i

  return max_iters

mandel_gpu = cuda.jit(restype=uint32, argtypes=[f8, f8, uint32], device=True)(mandel)

@cuda.jit(argtypes=[f8, f8, f8, f8, uint8[:,:], uint32])
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  startX, startY = cuda.grid(2)
  gridX = cuda.gridDim.x * cuda.blockDim.x;
  gridY = cuda.gridDim.y * cuda.blockDim.y;

  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y 
      image[y, x] = mandel_gpu(real, imag, iters)



"""
gimage = np.zeros((1024, 1536), dtype = np.uint8)
blockdim = (32, 8)
griddim = (32,16)

start = timer()
d_image = cuda.to_device(gimage)
mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20) 
d_image.to_host()
dt = timer() - start

print ("Mandelbrot created on GPU in %f s" % dt)

imshow(gimage)
show()

"""

string1 = "abcdefgh"
string2 = "abcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcba"

values = np.zeros((len(string2)-len(string1)+1), dtype = np.uint8)

start = timer()
leven_kernel(string1, string2, values)
dt = timer() - start

print (values, '\n', dt)
