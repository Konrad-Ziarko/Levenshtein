using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Zniffer;
using Zniffer.Levenshtein;


namespace CustomExtensions {


    public static class ExtensionMethods {
        public static int Map(this int value, int fromSource, int toSource, int fromTarget = 0, int toTarget = 255) {
            return (int)((float)(value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget);
        }
    }
    public static class StringExtension {

        #region CUDA

        private static LevenshteinMatches LevenshteinSingleMatrixGPU(string str, int[] A, int[] B, int[,] AB) {

            //levenshtein
            return null;
        }


        #endregion


        public static LevenshteinMatches Levenshtein(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false, LevenshteinMode mode = 0) {
            str = System.Text.RegularExpressions.Regex.Replace(str, @"\p{C}+", string.Empty);//remove nonprintable characters
            str = str.Replace("\0", "");

            if (str == null || str.Length == 0 || expression == null || expression.Length == 0)
                return new LevenshteinMatches();

            if (mode == LevenshteinMode.SingleMatixCPU) {
                return str.LevenshteinSingleMatrixCPU(expression, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.SplitForSingleMatrixCPU) {//splits words
                return str.LevenshteinSplitForSingleMatrixCPU(expression, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.MultiMatrixSingleThreadCPU) {
                return str.LevenshteinMultiMatrixSingleThread(expression, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.MultiMatrixParallelCPU) {
                return str.LevenshteinMultiMatrixParallel(expression, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.DualRowCPU) {//memory efficient
                return str.LevenshteinDualRowCPU(expression, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.ThreeDimMatrixCPU) {
                return str.LevenshteinThreeDimMatrixCPU(expression, maxDistance, onlyBestResults, caseSensitive);
            }
            else {
                throw new NotImplementedException();
            }
        }

        #region MutliMatrix
        public static LevenshteinMatches LevenshteinMultiMatrixSingleThread(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = expression.Length / 2;
            int exprLen = expression.Length;
            long strLen = str.Length - exprLen + 1;
            int[] results = new int[strLen];
            int[,] dimension = new int[exprLen + 1, exprLen + 1];

            for (int i = 0; i < strLen; i++) {
                results[i] = SqueareLevenshteinCPU(dimension, str.Substring(i, exprLen), expression, caseSensitive);
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


        public static LevenshteinMatches LevenshteinMultiMatrixParallel(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {

            if (maxDistance < 0)
                maxDistance = expression.Length / 2;

            //check if system will run out of memory, if so split string/ .NET 2 GB object size limit 
            var a = expression.Length * expression.Length * str.Length;
            if (a > 15000000) {
                int mid = str.Length / 2;
                string leftPart = str.Substring(0, mid + expression.Length - 1);
                string rightPart = str.Substring(mid);

                return new LevenshteinMatches(leftPart.LevenshteinMultiMatrixParallel(expression, maxDistance, onlyBestResults, caseSensitive), rightPart.LevenshteinMultiMatrixParallel(expression, maxDistance, onlyBestResults, caseSensitive));
            }

            int exprLen = expression.Length;
            int strLen = str.Length;
            int numOfMatrixes = strLen - exprLen + 1;
            int[] results = new int[strLen];
            int[][,] dimension = new int[strLen][,];
            for (int i = 0; i < numOfMatrixes; i++) {
                dimension[i] = new int[exprLen + 1, exprLen + 1];
            }

            Parallel.For(0, numOfMatrixes, i => {
                results[i] = RectangleLevenshteinCPU(dimension[i], str.Substring(i, exprLen), expression, caseSensitive);
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

        public static string GetContext(this string str, ref int position, int length, int paddingLength = 10) {
            int startPosition = position >= paddingLength ? position - paddingLength : 0;
            int endPosition = str.Length;

            string tmp = str;
            int tmpPosition = position;

            if (position + length + paddingLength <= str.Length)
                endPosition = position + length + paddingLength;
            try {
                tmp = tmp.Substring(startPosition, position - startPosition) + "<" + MainWindow.COLORTAG + ">" + tmp.Substring(position, length) + "</" + MainWindow.COLORTAG + ">" + tmp.Substring(position + length, endPosition - (position + length));
            }
            catch { };
            if (position > paddingLength)
                position = position - startPosition;
            return tmp;
        }

        #region SingleMatrix

        public static LevenshteinMatches LevenshteinThreeDimMatrixCPU(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = expression.Length / 2;
            int strLen = str.Length;
            int exprLen = expression.Length;
            if (strLen == 0 || exprLen == 0)
                return new LevenshteinMatches();

            if(strLen < exprLen) {
                str.Swap(expression);
            }

            int[,,] dimension = new int[strLen + 2 - exprLen, exprLen + 1, exprLen + 1];


            if (!caseSensitive) {
                str = str.ToUpper();
                expression = expression.ToUpper();
            }

            for (int i = 0; i < strLen + 2 - exprLen; i++) {
                for (int j = 0; j <= exprLen; j++) {
                    dimension[i, 0, j] = j;
                    dimension[i, j, 0] = j;
                }
            }
            for (int k = 0; k < strLen + 2 - exprLen; k++) {
                for (int i = 1; i <= exprLen; i++) {
                    for (int j = 1; j <= exprLen; j++) {
                        if (j - 1 + k < exprLen && str[i - 1] == expression[j - 1 + k]) {
                            dimension[k, i, j] = dimension[k, i - 1, j - 1];//if characters are same copy diagonal value
                        }
                        else {
                            dimension[k, i, j] = Math.Min(Math.Min(dimension[k, i - 1, j], dimension[k, i, j - 1]), dimension[k, i - 1, j - 1]) + 1;
                        }
                    }
                }
            }
            List<LevenshteinMatch> newMatches = new List<LevenshteinMatch>();

            for (int i = 0; i < strLen + 2 - exprLen; i++) {
                if (maxDistance >= dimension[i, exprLen, exprLen]) {
                    newMatches.Add(new LevenshteinMatch(str, i, exprLen, dimension[i, exprLen, exprLen]));
                }
            }
            return new LevenshteinMatches(newMatches);
        }

        public static LevenshteinMatches LevenshteinDualRowCPU(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = expression.Length / 2;

            int strLen = str.Length;
            int exprLen = expression.Length;
            if (strLen == 0 || exprLen == 0)
                return null;

            int[,] dimension = new int[2, exprLen + 1];

            //Matrix not even
            int distance = DualRowsLevenshteinCPU(dimension, str, expression, caseSensitive);
            if (distance <= maxDistance)
                return new LevenshteinMatches(new LevenshteinMatch(str, 0, strLen, distance));
            else
                return null;
        }

        public static LevenshteinMatches LevenshteinSplitForSingleMatrixCPU(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = expression.Length / 2;

            int maxWordsLengthDiff = maxDistance;

            List<string> words = str.Split(new[] { '\n', '\r', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries).ToList();
            for (int i = words.Count - 1; i >= 0; i--) {
                //remove too long and too short words
                if (words[i].Length + maxWordsLengthDiff < expression.Length || words[i].Length - maxWordsLengthDiff > expression.Length) {//dist - 2
                    words.RemoveAt(i);
                }
            }
            List<LevenshteinMatch> newMatches = new List<LevenshteinMatch>();
            foreach (string word in words) {
                int strLen = word.Length;
                int exprLen = expression.Length;
                int[,] dimension = new int[strLen + 1, exprLen + 1];

                if (strLen == exprLen) {
                    //if matrix is square
                    int distance = SqueareLevenshteinCPU(dimension, word, expression, caseSensitive);
                    if (distance <= maxDistance)
                        newMatches.Add(new LevenshteinMatch(str, str.IndexOf(word), strLen, distance));
                }
                else {
                    //Matrix not even
                    int distance = RectangleLevenshteinCPU(dimension, word, expression, caseSensitive);
                    if (distance <= maxDistance)
                        newMatches.Add(new LevenshteinMatch(str, str.IndexOf(word), strLen, distance));
                }
            }
            return new LevenshteinMatches(newMatches);
        }

        public static LevenshteinMatches LevenshteinSingleMatrixCPU(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = expression.Length / 2;

            //max str length 200000000 - due to .NET 2 GB, object size limitation
            int strLen = str.Length;
            int exprLen = expression.Length;
            if (strLen == 0 || exprLen == 0)
                return null;

            int[,] dimension = new int[strLen + 1, exprLen + 1];

            if (strLen == exprLen) {
                //if matrix is square
                int distance = SqueareLevenshteinCPU(dimension, str, expression, caseSensitive);
                if (distance <= maxDistance)
                    return new LevenshteinMatches(new LevenshteinMatch(str, 0, strLen, distance));
                else
                    return null;
            }
            else {
                //Matrix not even
                int distance = RectangleLevenshteinCPU(dimension, str, expression, caseSensitive);
                if (dimension[strLen, exprLen] <= maxDistance)
                    return new LevenshteinMatches(new LevenshteinMatch(str, 0, strLen, dimension[strLen, exprLen]));
                else
                    return null;
            }
        }
        #endregion

        //Wagnera–Fischera dynamic programming
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
        public static int DualRowsLevenshteinCPU(int[,] arr, string str1, string str2, bool caseSensitive = false) {
            if (!caseSensitive) {
                str1 = str1.ToUpper();
                str2 = str2.ToUpper();
            }

            for (int i = 0; i <= str2.Length; i++) {
                arr[0, i] = i;
            }
            arr[1, 0] = 1;

            for (int i = 1; i <= str1.Length; i++) {
                for (int j = 1; j <= str2.Length; j++) {
                    if (str1[i - 1] == str2[j - 1]) {
                        arr[1, j] = arr[0, j - 1];//if characters are same copy diagonal value
                    }
                    else {
                        arr[1, j] = Math.Min(Math.Min(arr[1, j - 1], arr[0, j]), arr[0, j - 1]) + 1;//if characters are diffrent compute min edit-distance
                    }
                }

                for (int k = 0; k <= str2.Length; k++) {
                    arr[0, k] = arr[1, k];
                    arr[1, k] = 0;
                }
                arr[1, 0] = i + 1;

            }
            return arr[0, str2.Length];//return min edit-distance
        }

        private static void Swap(this string str1, string str2) {
            string tmp = str1;
            str1 = str2;
            str2 = tmp;
        }
    }
}
