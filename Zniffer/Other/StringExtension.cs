
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Zniffer;

namespace CustomExtensions {
    public enum LevenshteinMode {
        SingleMatixCPU=0,
        SplitForSingleMatrixCPU,
        MultiMatrixSingleThread,
        MultiMatrixParallel,
    }

    public static class ExtensionMethods {
        public static int Map(this int value, int fromSource, int toSource, int fromTarget =0, int toTarget =255) {
            return (int)((float)(value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget);
        }
    }
    public static class StringExtension {

        #region CUDA


        #endregion


        public static LevenshteinMatches Levenshtein(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false, LevenshteinMode mode = 0) {
            if (mode == LevenshteinMode.SingleMatixCPU) {
                return str.LevenshteinSingleMatrixCPU(expression, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.SplitForSingleMatrixCPU) {
                return null;
            }
            else if(mode == LevenshteinMode.MultiMatrixSingleThread) {
                return str.LevenshteinMultiMatrixSingleThread(expression, maxDistance, onlyBestResults, caseSensitive);
            }
            else if (mode == LevenshteinMode.MultiMatrixParallel) {
                return str.LevenshteinMultiMatrixParallel(expression, maxDistance, onlyBestResults, caseSensitive);
            }
            else {
                throw new NotImplementedException();
            }
        }

        #region MutliMatrix
        private static LevenshteinMatches LevenshteinMultiMatrixSingleThread(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = expression.Length/2;

            int exprLen = expression.Length;
            long strLen = str.Length - exprLen + 1;
            int[] results = new int[strLen];
            int[,] dimension = new int[exprLen + 1, exprLen + 1];

            for (int i = 0; i < strLen; i++) {
                results[i] = SqueareLevenshteinCPU(dimension, str.Substring(i, exprLen), expression, exprLen, caseSensitive);
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


        private static LevenshteinMatches LevenshteinMultiMatrixParallel(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (str.Length > 0) {
                if (maxDistance < 0)
                    maxDistance = expression.Length / 2;

                //check if system will run out of memory, if so split string
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
                    results[i] = SqueareLevenshteinCPU(dimension[i], str.Substring(i, exprLen), expression, exprLen, caseSensitive);
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
            return new LevenshteinMatches();
        }

        #endregion


        public static string GetContext(this string str, ref int position, int length, int paddingLength = 10) {
            int startPosition = position >= paddingLength ? position - paddingLength : 0;
            int endPosition = str.Length;

            string tmp = str;
            int tmpPosition = position;

            if (position + length + paddingLength <= str.Length)
                endPosition = position + length + paddingLength;

            tmp = tmp.Substring(startPosition, position - startPosition) + "<" + MainWindow.COLORTAG + ">" + tmp.Substring(position, length) +"</" + MainWindow.COLORTAG + ">" + tmp.Substring(position + length, endPosition - (position + length));
            if (position > paddingLength)
                position = position - startPosition;
            return tmp;
        }

        #region SingleMatrix
        private static LevenshteinMatches LevenshteinSplitForSingleMatrixCPU(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            return null;
        }

        private static LevenshteinMatches LevenshteinSingleMatrixCPU(this string str, string expression, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            if (maxDistance < 0)
                maxDistance = expression.Length/2;

            //max str length 200000000
            int strLen = str.Length;
            int exprLen = expression.Length;
            if (strLen == 0 || exprLen == 0)
                return null;

            int[,] dimension = new int[strLen + 1, exprLen + 1];
            int matchCost = 0;


            if (strLen == exprLen) {
                //if matrix is square compute single matrix
                int distance = SqueareLevenshteinCPU(dimension, str, expression, strLen, caseSensitive);
                if (distance < maxDistance)
                    return new LevenshteinMatches(new LevenshteinMatch(str, 0, strLen, distance));//temp
                else
                    return null;
            }
            else {
                //Matrix initialization
                for (int i = 0; i <= strLen; i++)
                    dimension[i, 0] = i;
                for (int j = 0; j <= exprLen; j++)
                    dimension[0, j] = j;
            }

            //fill out matrix - Levenshtein algorithm
            for (int j = 1; j <= exprLen; j++) {
                for (int i = 1; i <= strLen; i++) {
                    if (str[i - 1] == expression[j - 1])
                        matchCost = 0;
                    else
                        matchCost = 1;

                    dimension[i, j] = Math.Min(Math.Min(dimension[i - 1, j] + 1, dimension[i, j - 1] + 1), dimension[i - 1, j - 1] + matchCost);
                }
            }

            //find best match in string

            if (dimension[strLen, exprLen] <= maxDistance)
                return new LevenshteinMatches(new LevenshteinMatch(str, 0, strLen, dimension[strLen, exprLen]));//temp
            else
                return null;
        }
        #endregion

        private static int SqueareLevenshteinCPU(int[,] arr, string str1, string str2, int len, bool caseSensitive = false) {
            for (int i = 0; i <= len; i++) {
                arr[i, 0] = i;
                arr[0, i] = i;
            }
            if (!caseSensitive) {
                str1 = str1.ToUpper();
                str2 = str2.ToUpper();
            }

            int matchCost = 0;
            for (int i = 1; i <= len; i++) {
                for (int j = 1; j <= len; j++) {
                    if (str1[i - 1] == str2[j - 1])
                        matchCost = 0;
                    else
                        matchCost = 1;

                    arr[i, j] = Math.Min(Math.Min(arr[i - 1, j] + 1, arr[i, j - 1] + 1), arr[i - 1, j - 1] + matchCost);
                }
            }
            return arr[len, len];
        }

    }
}
