from numba import cuda
from numba import *
import numpy as np
from timeit import default_timer as timer

def leven_dist(A, B):
    h = len(A)+1
    w = len(B)+1
    metric = [[0 for x in range(w)] for y in range(h)] 
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
    return metric[len(A)][len(B)]      

def main():
    string1 = "123456"
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

    start = timer()
    metric = leven_dist(A,B)
    dt = timer() - start

    print (metric, '\n', dt)

    
if __name__ == '__main__':
    main()