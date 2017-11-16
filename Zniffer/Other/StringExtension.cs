
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

        
        public static LevenshteinMatches LevenshteinSingleThread(this string str, string expression, int maxDistance) {
            if (str.Length > expression.Length + 1) {
                int len = expression.Length;
                int strLen = str.Length - len + 1;
                int[] results = new int[strLen];
                int[][,] dimension = new int[strLen][,];
                for (int i = 0; i < strLen; i++) {
                    dimension[i] = new int[len + 1, len + 1];
                }

                string source = str;
                source = source.ToUpper();
                expression = expression.ToUpper();

                Parallel.For(0, strLen, i => {
                    results[i] = SqueareLevenshtein(dimension[i], str.Substring(i, len).ToUpper(), expression, len);
                });

                LevenshteinMatches matches = new LevenshteinMatches();

                
                Parallel.For(0, strLen, i => {
                    if (results[i] <= maxDistance) {
                        lock (matches)
                            matches.addMatch(str.Substring(i, len), i, len, results[i]);
                    }
                });

                return matches;
            }
            else {
                LevenshteinMatch match = str.LevenshteinCPU(expression, maxDistance);
                if (match != null)
                    return new LevenshteinMatches(match);
                else
                    return new LevenshteinMatches();
            }
        }


        #region SingleMatrix

        public static LevenshteinMatch LevenshteinCPU(this string str, string expression, int maxDistance) {
            string source = str;
            if (source.Length == 0 || expression.Length == 0)
                return null;

            source = source.ToUpper();
            expression = expression.ToUpper();

            int[,] dimension = new int[source.Length + 1, expression.Length + 1];
            int matchCost = 0;


            if (source.Length == expression.Length)
                return str.SquareLevenshteinCPU(expression, maxDistance);
            else {
                for (int i = 0; i <= source.Length; i++)
                    dimension[i, 0] = i;

                for (int j = 0; j <= expression.Length; j++)
                    dimension[0, j] = j;
            }


            for (int i = 1; i <= source.Length; i++) {
                for (int j = 1; j <= expression.Length; j++) {
                    if (source[i - 1] == expression[j - 1])
                        matchCost = 0;
                    else
                        matchCost = 1;

                    dimension[i, j] = Math.Min(Math.Min(dimension[i - 1, j] + 1, dimension[i, j - 1] + 1), dimension[i - 1, j - 1] + matchCost);
                }
            }

            if (dimension[source.Length, expression.Length] <= maxDistance)
                return new LevenshteinMatch(str, 0, source.Length, dimension[source.Length, expression.Length]);
            else
                return null;
        }

        public static LevenshteinMatch SquareLevenshteinCPU(this string str, string expression, int maxDistance) {
            string source = str;
            if (source.Length == 0 || expression.Length == 0)
                return null;

            source = source.ToUpper();
            expression = expression.ToUpper();

            int len = source.Length;

            int[,] dimension = new int[len + 1, len + 1];

            SqueareLevenshtein(dimension, source, expression, len);

            if (dimension[len, len] <= maxDistance)
                return new LevenshteinMatch(str, 0, len, dimension[len, len]);
            else
                return null;
        }

        public static int SqueareLevenshtein(int[,] arr, string str1, string str2, int len) {
            for (int i = 0; i <= len; i++) {
                arr[i, 0] = i;
                arr[0, i] = i;
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
