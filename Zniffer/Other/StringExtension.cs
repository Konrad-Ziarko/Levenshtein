using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Zniffer.Levenshtein;
using Zniffer.Other;

namespace CustomExtensions {


    public static class ExtensionMethods {
        public static int Map(this int value, int fromSource, int toSource, int fromTarget = 0, int toTarget = 255) {
            return (int)((float)(value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget);
        }
    }
    public static class StringExtension {
        static LevenshteinGPU gpu = LevenshteinGPU.instance;
        public static LevenshteinMatches Levenshtein(this string str, string pattern, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false, LevenshteinMode mode = 0) {
            str = System.Text.RegularExpressions.Regex.Replace(str, @"\p{C}+", string.Empty);//remove nonprintable characters
            str = str.Replace("\0", "");

            if (str == null || str.Length == 0 || pattern == null || pattern.Length == 0)
                return new LevenshteinMatches();

            if (mode == LevenshteinMode.SingleMatixCPU) {
                return str.LevenshteinSingleMatrixCPU(pattern, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.SplitForSingleMatrixCPU) {//splits words
                return str.LevenshteinSplitForSingleMatrixCPU(pattern, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.MultiMatrixSingleThreadCPU) {
                return str.LevenshteinMultiMatrixSingleThread(pattern, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.MultiMatrixParallelCPU) {
                return str.LevenshteinMultiMatrixParallel(pattern, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.SplitDualRowCPU) {//memory efficient
                return str.LevenshteinDualRowCPU(pattern, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.ThreeDimMatrixCPU) {
                return str.LevenshteinThreeDimMatrixCPU(pattern, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.ThreeDimMatrixGPU) {
                return gpu.LevenshteinSingleMatrixGPU(str, pattern, maxDistance, onlyBestResults, caseSensitive);
                //return str.LevenshteinSingleMatrixGPU(pattern, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.ThreeDimMatrixParallelCPU) {
                return str.LevenshteinThreeDimMatrixParallelCPU(pattern, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.SplitForParallelCPU) {
                return str.LevenshteinSplitForParallelCPU(pattern, maxDistance, onlyBestResults, caseSensitive);
            }
            else {
                throw new NotImplementedException();
            }
        }

        #region MutliMatrix
        public static LevenshteinMatches LevenshteinMultiMatrixSingleThread(this string str, string pattern, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = pattern.Length / 2;
            int exprLen = pattern.Length;
            long strLen = str.Length - exprLen + 1;
            int[] results = new int[strLen];
            int[,] dimension = new int[exprLen + 1, exprLen + 1];

            for (int i = 0; i < strLen; i++) {
                results[i] = SqueareLevenshteinCPU(dimension, str.Substring(i, exprLen), pattern, caseSensitive);
            }

            LevenshteinMatches matches = new LevenshteinMatches();

            int bestDistance = int.MaxValue;

            for (int i = 0; i < strLen; i++) {
                if (results[i] <= maxDistance) {
                    bestDistance = bestDistance < results[i] ? bestDistance : results[i];
                    matches.addMatch(str, i, exprLen, results[i]);
                }
            }

            if (onlyBestResults) {
                matches.removeMatches(bestDistance);
            }

            return matches;

        }


        public static LevenshteinMatches LevenshteinMultiMatrixParallel(this string str, string pattern, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {

            if (maxDistance < 0)
                maxDistance = pattern.Length / 2;

            //check if system will run out of memory, if so split string/ .NET 2 GB object size limit 
            var a = pattern.Length * pattern.Length * str.Length;
            if (a > 15000000) {
                int mid = str.Length / 2;
                string leftPart = str.Substring(0, mid + pattern.Length - 1);
                string rightPart = str.Substring(mid);

                return new LevenshteinMatches(leftPart.LevenshteinMultiMatrixParallel(pattern, maxDistance, onlyBestResults, caseSensitive), rightPart.LevenshteinMultiMatrixParallel(pattern, maxDistance, onlyBestResults, caseSensitive));
            }

            int exprLen = pattern.Length;
            int strLen = str.Length;
            int numOfMatrixes = strLen - exprLen + 1;
            int[] results = new int[strLen];
            int[][,] dimension = new int[strLen][,];
            for (int i = 0; i < numOfMatrixes; i++) {
                dimension[i] = new int[exprLen + 1, exprLen + 1];
            }

            Parallel.For(0, numOfMatrixes, i => {
                results[i] = RectangleLevenshteinCPU(dimension[i], str.Substring(i, exprLen), pattern, caseSensitive);
            });

            LevenshteinMatches matches = new LevenshteinMatches();

            object bestDistance = int.MaxValue;

            Parallel.For(0, numOfMatrixes, i => {
                if (results[i] <= maxDistance) {
                    lock (matches) {
                        //lock (bestDistance)
                        bestDistance = (int)bestDistance < results[i] ? bestDistance : results[i];
                        matches.addMatch(str, i, exprLen, results[i]);
                    }
                }
            });

            if (onlyBestResults) {
                matches.removeMatches((int)bestDistance);
            }

            return matches;
        }
        #endregion

        #region SingleMatrix

        public static LevenshteinMatches LevenshteinThreeDimMatrixParallelCPU(this string originalString, string pattern, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = pattern.Length / 2;
            string str = originalString;
            if (originalString.Length < pattern.Length)
                return new LevenshteinMatches();

            int strLen = str.Length;
            int patternLen = pattern.Length;
            if (strLen == 0 || patternLen == 0)
                return new LevenshteinMatches();

            int compareLength = patternLen;
            int firstDim = strLen + 2 - patternLen;

            int[,,] dimension = new int[firstDim, compareLength + 1, compareLength + 1];


            if (!caseSensitive) {
                str = str.ToUpper();
                pattern = pattern.ToUpper();
            }

            Parallel.For(0, firstDim, i => {
                for (int j = 0; j <= compareLength; j++) {
                    dimension[i, 0, j] = j;
                    dimension[i, j, 0] = j;
                }
            });


            Parallel.For(0, firstDim, k => {
                for (int i = 1; i <= compareLength; i++) {
                    for (int j = 1; j <= compareLength; j++) {
                        if (k + i - 1 < strLen && str[k + i - 1] == pattern[j - 1]) {
                            dimension[k, i, j] = dimension[k, i - 1, j - 1];//if characters are same copy diagonal value
                        }
                        else {
                            dimension[k, i, j] = Math.Min(Math.Min(dimension[k, i - 1, j], dimension[k, i, j - 1]), dimension[k, i - 1, j - 1]) + 1;
                        }
                    }
                }
            });
            List<LevenshteinMatch> newMatches = new List<LevenshteinMatch>();

            for (int i = 0; i < firstDim; i++) {
                if (maxDistance >= dimension[i, compareLength, compareLength]) {
                    newMatches.Add(new LevenshteinMatch(originalString, i, compareLength, dimension[i, compareLength, compareLength]));
                }
            }
            return new LevenshteinMatches(newMatches);
        }

        public static LevenshteinMatches LevenshteinThreeDimMatrixCPU(this string originalString, string pattern, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = pattern.Length / 2;
            string str = originalString;
            if (originalString.Length < pattern.Length)
                return new LevenshteinMatches();

            int strLen = str.Length;
            int patternLen = pattern.Length;
            if (strLen == 0 || patternLen == 0)
                return new LevenshteinMatches();

            int compareLength = patternLen;
            int firstDim = strLen + 2 - patternLen;

            int[,,] dimension = new int[firstDim, compareLength + 1, compareLength + 1];


            if (!caseSensitive) {
                str = str.ToUpper();
                pattern = pattern.ToUpper();
            }

            for (int i = 0; i < firstDim; i++) {
                for (int j = 0; j <= compareLength; j++) {
                    dimension[i, 0, j] = j;
                    dimension[i, j, 0] = j;
                }
            }
            for (int k = 0; k < firstDim; k++) {
                for (int i = 1; i <= compareLength; i++) {
                    for (int j = 1; j <= compareLength; j++) {
                        if (k + i - 1 < strLen && str[k + i - 1] == pattern[j - 1]) {
                            dimension[k, i, j] = dimension[k, i - 1, j - 1];//if characters are same copy diagonal value
                        }
                        else {
                            dimension[k, i, j] = Math.Min(Math.Min(dimension[k, i - 1, j], dimension[k, i, j - 1]), dimension[k, i - 1, j - 1]) + 1;
                        }
                    }
                }
            }
            List<LevenshteinMatch> newMatches = new List<LevenshteinMatch>();

            for (int i = 0; i < firstDim; i++) {
                if (maxDistance >= dimension[i, compareLength, compareLength]) {
                    newMatches.Add(new LevenshteinMatch(originalString, i, compareLength, dimension[i, compareLength, compareLength]));
                }
            }
            return new LevenshteinMatches(newMatches);
        }

        public static LevenshteinMatches LevenshteinDualRowCPU(this string str, string pattern, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = pattern.Length / 2;

            int maxWordsLengthDiff = maxDistance;

            List<string> words = str.Split(new[] { '\n', '\r', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries).ToList();
            for (int i = words.Count - 1; i >= 0; i--) {
                //remove too long and too short words
                if (words[i].Length + maxWordsLengthDiff < pattern.Length || words[i].Length - maxWordsLengthDiff > pattern.Length) {//dist - 2
                    words.RemoveAt(i);
                }
            }
            List<LevenshteinMatch> newMatches = new List<LevenshteinMatch>();
            foreach (string word in words) {
                int strLen = word.Length;
                int exprLen = pattern.Length;
                int[] firstRow = new int[exprLen + 1];
                int[] secondRow = new int[exprLen + 1];


                int distance = DualRowsLevenshteinCPU(firstRow, secondRow, word, pattern, caseSensitive);
                if (distance <= maxDistance)
                    newMatches.Add(new LevenshteinMatch(str, str.IndexOf(word), strLen, distance));
            }
            return new LevenshteinMatches(newMatches);
        }



        public static LevenshteinMatches LevenshteinSplitForParallelCPU(this string str, string pattern, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = pattern.Length / 2;

            int maxWordsLengthDiff = maxDistance;

            List<string> words = str.Split(new[] { '\n', '\r', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries).ToList();
            for (int i = words.Count - 1; i >= 0; i--) {
                //remove too long and too short words
                if (words[i].Length + maxWordsLengthDiff < pattern.Length || words[i].Length - maxWordsLengthDiff > pattern.Length) {//dist - 2
                    words.RemoveAt(i);
                }
            }
            List<LevenshteinMatch> newMatches = new List<LevenshteinMatch>();

            int exprLen = pattern.Length;

            Parallel.For(0, words.Count, idx => {
                string word = words[idx];
                int strLen = word.Length;
                
                int[,] dimension = new int[strLen + 1, exprLen + 1];

                if (strLen == exprLen) {
                    //if matrix is square
                    int distance = SqueareLevenshteinCPU(dimension, word, pattern, caseSensitive);
                    if (distance <= maxDistance)
                        newMatches.Add(new LevenshteinMatch(str, str.IndexOf(word), strLen, distance));
                }
                else {
                    //Matrix not even
                    int distance = RectangleLevenshteinCPU(dimension, word, pattern, caseSensitive);
                    if (distance <= maxDistance)
                        newMatches.Add(new LevenshteinMatch(str, str.IndexOf(word), strLen, distance));
                }
            });
            return new LevenshteinMatches(newMatches);
        }

        public static LevenshteinMatches LevenshteinSplitForSingleMatrixCPU(this string str, string pattern, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = pattern.Length / 2;

            int maxWordsLengthDiff = maxDistance;

            List<string> words = str.Split(new[] { '\n', '\r', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries).ToList();
            for (int i = words.Count - 1; i >= 0; i--) {
                //remove too long and too short words
                if (words[i].Length + maxWordsLengthDiff < pattern.Length || words[i].Length - maxWordsLengthDiff > pattern.Length) {//dist - 2
                    words.RemoveAt(i);
                }
            }
            List<LevenshteinMatch> newMatches = new List<LevenshteinMatch>();
            foreach (string word in words) {
                int strLen = word.Length;
                int exprLen = pattern.Length;
                int[,] dimension = new int[strLen + 1, exprLen + 1];

                if (strLen == exprLen) {
                    //if matrix is square
                    int distance = SqueareLevenshteinCPU(dimension, word, pattern, caseSensitive);
                    if (distance <= maxDistance)
                        newMatches.Add(new LevenshteinMatch(str, str.IndexOf(word), strLen, distance));
                }
                else {
                    //Matrix not even
                    int distance = RectangleLevenshteinCPU(dimension, word, pattern, caseSensitive);
                    if (distance <= maxDistance)
                        newMatches.Add(new LevenshteinMatch(str, str.IndexOf(word), strLen, distance));
                }
            }
            return new LevenshteinMatches(newMatches);
        }

        public static LevenshteinMatches LevenshteinSingleMatrixCPU(this string str, string pattern, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = pattern.Length / 2;

            //max str length 200000000 - due to .NET 2 GB, object size limitation
            int strLen = str.Length;
            int exprLen = pattern.Length;
            if (strLen == 0 || exprLen == 0)
                return null;

            int[,] dimension = new int[strLen + 1, exprLen + 1];

            if (strLen == exprLen) {
                //if matrix is square
                int distance = SqueareLevenshteinCPU(dimension, str, pattern, caseSensitive);
                if (distance <= maxDistance)
                    return new LevenshteinMatches(new LevenshteinMatch(str, 0, strLen, distance));
                else
                    return null;
            }
            else {
                //Matrix not even
                int distance = RectangleLevenshteinCPU(dimension, str, pattern, caseSensitive);
                if (dimension[strLen, exprLen] <= maxDistance)
                    return new LevenshteinMatches(new LevenshteinMatch(str, 0, strLen, dimension[strLen, exprLen]));
                else
                    return null;
            }
        }
        #endregion

        //Wagner–Fischer dynamic programming
        public static int SqueareLevenshteinCPU(int[,] arr, string str1, string str2, bool caseSensitive = false) {
            int len = str1.Length;
            for (int i = 0; i <= len; i++) {
                arr[i, 0] = i;
                arr[0, i] = i;
            }
            if (!caseSensitive) {
                str1 = str1.ToUpper();
                str2 = str2.ToUpper();
            }

            for (int i = 1; i <= len; i++) {
                for (int j = 1; j <= len; j++) {
                    if (str1[i - 1] == str2[j - 1]) {
                        arr[i, j] = arr[i - 1, j - 1];//if characters are same copy diagonal value
                    }
                    else {
                        arr[i, j] = Math.Min(Math.Min(arr[i - 1, j], arr[i, j - 1]), arr[i - 1, j - 1]) + 1;//if characters are diffrent compute min edit-distance
                    }
                }
            }
            return arr[len, len];//return min edit-distance
        }
        public static int RectangleLevenshteinCPU(int[,] arr, string str1, string str2, bool caseSensitive = false) {
            for (int i = 0; i <= str1.Length; i++) {
                arr[i, 0] = i;
            }
            for (int i = 1; i <= str2.Length; i++) {
                arr[0, i] = i;
            }

            if (!caseSensitive) {
                str1 = str1.ToUpper();
                str2 = str2.ToUpper();
            }

            for (int i = 1; i <= str1.Length; i++) {
                for (int j = 1; j <= str2.Length; j++) {
                    if (str1[i - 1] == str2[j - 1]) {
                        arr[i, j] = arr[i - 1, j - 1];//if characters are same copy diagonal value
                    }
                    else {
                        arr[i, j] = Math.Min(Math.Min(arr[i - 1, j], arr[i, j - 1]), arr[i - 1, j - 1]) + 1;//if characters are diffrent compute min edit-distance
                    }
                }
            }
            return arr[str1.Length, str2.Length];//return min edit-distance
        }
        public static int DualRowsLevenshteinCPU(int[] firstRow, int[] secondRow, string str1, string str2, bool caseSensitive = false) {
            if (!caseSensitive) {
                str1 = str1.ToUpper();
                str2 = str2.ToUpper();
            }

            for (int i = 0; i <= str2.Length; i++) {
                firstRow[i] = i;
            }
            secondRow[0] = 1;

            for (int i = 1; i <= str1.Length; i++) {
                for (int j = 1; j <= str2.Length; j++) {
                    if (str1[i - 1] == str2[j - 1]) {
                        secondRow[j] = firstRow[j - 1];//if characters are same copy diagonal value
                    }
                    else {
                        secondRow[j] = Math.Min(Math.Min(secondRow[j - 1], firstRow[j]), firstRow[j - 1]) + 1;//if characters are diffrent compute min edit-distance
                    }
                }
                firstRow = secondRow;
                secondRow = new int[str2.Length + 1];
                secondRow[0] = i + 1;
            }
            return firstRow[str2.Length];//return min edit-distance
        }
    }
}
