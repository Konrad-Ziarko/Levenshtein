from numba import cuda
from numba import *
import numpy as np
from timeit import default_timer as timer

@autojit
def leven_dist(A, B, w1, w2):
    metric = np.zeros((w1, w1), dtype = np.uint32)
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

@autojit
def leven_jit(word, line, metric_values):
    wordLen = len(word)
    maxPos = len(line)-wordLen+1
    for i in range(maxPos):
        metric_values[i] = leven_dist(word, line[i:i+wordLen], wordLen+1, wordLen)


def main():
    string1 = "123456"
    string2 = "abcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcbaabcdefghijklmnoprstuvwxyzzyxwvutsrponmlkjihgfedcba"
    string2 = string2*5


    list1 = []
    list2 = []
    for i in string1:
        list1.append(ord(i))
    for i in string2:
        list2.append(ord(i))

    A = np.array(list1, dtype=np.uint32)
    B = np.array(list2, dtype=np.uint32)

    #jit
    values = np.zeros((len(B)-len(A)+1), dtype = np.uint32)

    start = timer()
    leven_jit(A, B, values)
    dt = timer() - start

    print (min(values), '\n', dt)


    
if __name__ == '__main__':
    main()