
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Zniffer;

namespace CustomExtensions {
    public static class StringExtension {

        #region CUDA


        #endregion


        public static LevenshteinMatches LevenshteinSingleThread(this string str, string expression, int maxDistance, bool onlyBestResults = false, bool caseSensitive = false) {
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
                    matches.addMatch(str.Substring(i, exprLen), i, exprLen, results[i]);
                }
            }

            if (onlyBestResults) {
                matches.removeMatches(bestDistance);
            }

            return matches;

        }


        public static LevenshteinMatches LevenshteinParallel(this string str, string expression, int maxDistance, bool onlyBestResults = false, bool caseSensitive = false) {
            int exprLen = expression.Length;
            int strLen = str.Length - exprLen + 1;
            int[] results = new int[strLen];
            int[][,] dimension = new int[strLen][,];
            for (int i = 0; i < strLen; i++) {
                dimension[i] = new int[exprLen + 1, exprLen + 1];
            }

            Parallel.For(0, strLen, i => {
                results[i] = SqueareLevenshteinCPU(dimension[i], str.Substring(i, exprLen), expression, exprLen, caseSensitive);
            });

            LevenshteinMatches matches = new LevenshteinMatches();

            object bestDistance = int.MaxValue;

            Parallel.For(0, strLen, i => {
                if (results[i] <= maxDistance) {
                    lock (matches) {
                        //lock (bestDistance)
                            bestDistance = (int)bestDistance < results[i] ? bestDistance : results[i];
                        matches.addMatch(str.Substring(i, exprLen), i, exprLen, results[i]);
                    }
                }
            });

            if (onlyBestResults) {
                matches.removeMatches((int)bestDistance);
            }

            return matches;
        }

        public static string GetContext(this string str, int position, int length, int paddingLength = 10, char emptyChar = ' ') {
            string tmp = str;
            int startPosition = position >= paddingLength ? position : paddingLength;
            if (position < paddingLength)
                tmp = new string(emptyChar, paddingLength - position) + tmp;
            if (position + length > str.Length - paddingLength)
                tmp = tmp + new string(emptyChar, paddingLength - (str.Length - (position + length)));

            return tmp.Substring(startPosition - paddingLength, length + paddingLength*2);
        }


        #region SingleMatrix

        public static LevenshteinMatch LevenshteinSingleMatrixCPU(this string str, string expression, int maxDistance, bool caseSensitive = false) {
            int strLen = str.Length;
            int exprLen = expression.Length;
            if (strLen == 0 || exprLen == 0)
                return null;

            int[,] dimension = new int[strLen + 1, exprLen + 1];
            int matchCost = 0;


            if (strLen == exprLen) {
                int distance = SqueareLevenshteinCPU(dimension, str, expression, strLen, caseSensitive);
                if (distance < maxDistance)
                    return new LevenshteinMatch(str, 0, strLen, distance);
                else
                    return null;
            }
            else {
                for (int i = 0; i <= strLen; i++)
                    dimension[i, 0] = i;

                for (int j = 0; j <= exprLen; j++)
                    dimension[0, j] = j;
            }


            for (int i = 1; i <= strLen; i++) {
                for (int j = 1; j <= exprLen; j++) {
                    if (str[i - 1] == expression[j - 1])
                        matchCost = 0;
                    else
                        matchCost = 1;

                    dimension[i, j] = Math.Min(Math.Min(dimension[i - 1, j] + 1, dimension[i, j - 1] + 1), dimension[i - 1, j - 1] + matchCost);
                }
            }

            if (dimension[strLen, exprLen] <= maxDistance)
                return new LevenshteinMatch(str, 0, strLen, dimension[strLen, exprLen]);
            else
                return null;
        }


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
        #endregion
    }
}
