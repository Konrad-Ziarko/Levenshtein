import numpy as np
from timeit import default_timer as timer
from numba import cuda, vectorize
from pycuda.compiler import SourceModule


#calculate Levenshtein distance
def calcLevDist(str1, str2):
    #print(str1,'\n%s'%str2)
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
    #return metric

class Histogram:
    def __init__(self):
        self.histValues = {}
    def addToHist(self, char):
        if self.histValues.get(char)==None:
            self.histValues[char] = 1
        else:
            self.histValues[char]=self.histValues[char]+1
    def removeFromHist(self, char):
        if self.histValues[char]==None:
            pass
        else:
            self.histValues[char]=self.histValues[char]-1
            if self.histValues[char]==0:
                del self.histValues[char]
    def printHist(self):
        print (self.histValues)
    def computeSimilarity(self, secondObj):
        metricValue = 0
        foundDiffs = {}
        #foreach hist ele in self find elements in secon and vice versa
        for x in self.histValues:
            if secondObj.histValues.get(x) != None:
                foundDiffs[x] = 1
                if secondObj.histValues.get(x) != self.histValues.get(x):
                    metricValue = metricValue+1
        for y in secondObj.histValues:
            if foundDiffs.get(y) == None:
                if secondObj.histValues.get(y) != self.histValues.get(y):
                    metricValue = metricValue+1
        return metricValue

#calculate generalized Levenshtein distance
def calcLevGeneralization(str1, str2):
    if len(str1) == len(str2):
        cost = 0;
        for i in range(0, len(str1)):
            if str1[i] != str2[i]:
                cost=cost+1
        return cost

#find best matches in text; dist is equal to number of replace operations
def findGeneralizedMatches(str1, str2):
    start = timer()
    if len(str1) >= len(str2):
        strlen = len(str2)
        offset = strlen
        minDist = len(str2)+1
        bestMatch = []
        #matchVal = []
        tmpStr = str1[0:strlen]

        for i in range (0, len(str1)-strlen+1):
            newDist = calcLevGeneralization(tmpStr, str2)
            if newDist <= minDist:
                #print(tmpStr) #print matched string
                minDist = newDist
                bestMatch.append([offset-strlen, minDist])
                #matchVal.append(minDist)

            tmpStr = tmpStr[1:]
            if offset < len(str1):
                tmpStr = tmpStr + str1[offset]
            offset = offset+1
        #print(matchVal)
        vectoradd_time = timer() - start
        print("Time:%f" % vectoradd_time)
        return bestMatch

def findMatches(str1, str2):
    start = timer()
    if len(str1) >= len(str2):
        strlen = len(str2)
        offset = strlen
        minDist = len(str2)+1
        bestMatch = []
        #matchVal = []
        tmpStr = str1[0:strlen]

        for i in range (0, len(str1)-strlen+1):
            newDist = calcLevDist(tmpStr, str2)
            if newDist <= minDist:
                #print(tmpStr) #print matched string
                minDist = newDist
                bestMatch.append([offset-strlen, minDist])
                #matchVal.append(minDist)

            tmpStr = tmpStr[1:]
            if offset < len(str1):
                tmpStr = tmpStr + str1[offset]
            offset = offset+1
        #print(matchVal)
        vectoradd_time = timer() - start
        print("Time:%f" % vectoradd_time)
        return bestMatch

def main():

    str1 = 'Wojskowa akademia techniczna im gen Jaroslawa Dąbrowskiego w Warszawie'
    str2 = 'warszawie'
    
    str3 = "Piesasd"
    str4 = "Biesdsa"

    hist = Histogram()
    hist2 = Histogram()
    for char in str3:
        hist.addToHist(char)

    for char in str4:
        hist2.addToHist(char)

    hist.printHist()
    hist2.printHist()
    print(hist.computeSimilarity(hist2))

    #print('[Idx, DistVal]', findMatches(str1, str2))

    """
    with open('data.txt', 'r') as myfile:
        data=myfile.read().replace('\n', ' ')
        print('[Idx, DistVal]', findMatches(data, str2))
        print('[Idx, DistVal]', findGeneralizedMatches(data, str2))
       """ 
    """
    hist = Histogram()
    hist.addToHist(5)
    hist.addToHist(5)
    hist.addToHist(6)
    hist.addToHist(5)
    hist.addToHist(8)
    hist.removeFromHist(8)
    hist.printHist()
    """

if __name__ == '__main__':
    main()