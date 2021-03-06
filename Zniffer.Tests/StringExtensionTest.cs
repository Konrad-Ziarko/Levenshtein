// <copyright file="StringExtensionTest.cs">Copyright ©  2017</copyright>
using System;
using CustomExtensions;
using Microsoft.Pex.Framework;
using Microsoft.Pex.Framework.Validation;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Zniffer.Levenshtein;

namespace CustomExtensions.Tests {
    /// <summary>This class contains parameterized unit tests for StringExtension</summary>
    [PexClass(typeof(StringExtension))]
    [PexAllowedExceptionFromTypeUnderTest(typeof(InvalidOperationException))]
    [PexAllowedExceptionFromTypeUnderTest(typeof(ArgumentException), AcceptExceptionSubtypes = true)]
    [TestClass]
    public partial class StringExtensionTest {
        /// <summary>Test stub for Levenshtein(String, String, Int32, Boolean, Boolean, LevenshteinMode)</summary>
        [PexMethod(MaxRunsWithoutNewTests = 200, MaxConstraintSolverTime = 2)]
        [PexAllowedException(typeof(ArgumentOutOfRangeException))]
        [PexAllowedException(typeof(IndexOutOfRangeException))]
        [PexAllowedException(typeof(OverflowException))]
        [PexAllowedException(typeof(NotImplementedException))]
        public LevenshteinMatches LevenshteinTest(
            string str,
            string expression,
            int maxDistance,
            bool onlyBestResults,
            bool caseSensitive,
            LevenshteinMode mode
        ) {
            LevenshteinMatches result = StringExtension.Levenshtein
                                            (str, expression, maxDistance, onlyBestResults, caseSensitive, mode);
            return result;
            // TODO: add assertions to method StringExtensionTest.LevenshteinTest(String, String, Int32, Boolean, Boolean, LevenshteinMode)
        }

        /// <summary>Test stub for LevenshteinMultiMatrixSingleThread(String, String, Int32, Boolean, Boolean)</summary>
        [PexMethod(MaxRunsWithoutNewTests = 200, MaxConditions = 1000)]
        [PexAllowedException(typeof(OverflowException))]
        public LevenshteinMatches LevenshteinMultiMatrixSingleThreadTest(
            string str,
            string expression,
            int maxDistance,
            bool onlyBestResults,
            bool caseSensitive
        ) {
            LevenshteinMatches result = StringExtension.LevenshteinMultiMatrixSingleThread
                                            (str, expression, maxDistance, onlyBestResults, caseSensitive);
            return result;
            // TODO: add assertions to method StringExtensionTest.LevenshteinMultiMatrixSingleThreadTest(String, String, Int32, Boolean, Boolean)
        }

        /// <summary>Test stub for LevenshteinMultiMatrixParallel(String, String, Int32, Boolean, Boolean)</summary>
        [PexMethod(MaxRunsWithoutNewTests = 200, MaxConstraintSolverTime = 2, Timeout = 240)]
        [PexAllowedException(typeof(ArgumentOutOfRangeException))]
        public LevenshteinMatches LevenshteinMultiMatrixParallelTest(
            string str,
            string expression,
            int maxDistance,
            bool onlyBestResults,
            bool caseSensitive
        ) {
            LevenshteinMatches result = StringExtension.LevenshteinMultiMatrixParallel
                                            (str, expression, maxDistance, onlyBestResults, caseSensitive);
            return result;
            // TODO: add assertions to method StringExtensionTest.LevenshteinMultiMatrixParallelTest(String, String, Int32, Boolean, Boolean)
        }

        /// <summary>Test stub for LevenshteinThreeDimMatrixCPU(String, String, Int32, Boolean, Boolean)</summary>
        [PexMethod(MaxRunsWithoutNewTests = 200)]
        [PexAllowedException(typeof(IndexOutOfRangeException))]
        [PexAllowedException(typeof(OverflowException))]
        public LevenshteinMatches LevenshteinThreeDimMatrixCPUTest(
            string str,
            string expression,
            int maxDistance,
            bool onlyBestResults,
            bool caseSensitive
        ) {
            LevenshteinMatches result = StringExtension.LevenshteinThreeDimMatrixCPU
                                            (str, expression, maxDistance, onlyBestResults, caseSensitive);
            return result;
            // TODO: add assertions to method StringExtensionTest.LevenshteinThreeDimMatrixCPUTest(String, String, Int32, Boolean, Boolean)
        }

        /// <summary>Test stub for LevenshteinDualRowCPU(String, String, Int32, Boolean, Boolean)</summary>
        [PexMethod(MaxRunsWithoutNewTests = 200)]
        public LevenshteinMatches LevenshteinDualRowCPUTest(
            string str,
            string expression,
            int maxDistance,
            bool onlyBestResults,
            bool caseSensitive
        ) {
            LevenshteinMatches result = StringExtension.LevenshteinDualRowCPU
                                            (str, expression, maxDistance, onlyBestResults, caseSensitive);
            return result;
            // TODO: add assertions to method StringExtensionTest.LevenshteinDualRowCPUTest(String, String, Int32, Boolean, Boolean)
        }

        /// <summary>Test stub for LevenshteinSplitForSingleMatrixCPU(String, String, Int32, Boolean, Boolean)</summary>
        [PexMethod(MaxRunsWithoutNewTests = 200)]
        public LevenshteinMatches LevenshteinSplitForSingleMatrixCPUTest(
            string str,
            string expression,
            int maxDistance,
            bool onlyBestResults,
            bool caseSensitive
        ) {
            LevenshteinMatches result = StringExtension.LevenshteinSplitForSingleMatrixCPU
                                            (str, expression, maxDistance, onlyBestResults, caseSensitive);
            return result;
            // TODO: add assertions to method StringExtensionTest.LevenshteinSplitForSingleMatrixCPUTest(String, String, Int32, Boolean, Boolean)
        }

        /// <summary>Test stub for LevenshteinSingleMatrixCPU(String, String, Int32, Boolean, Boolean)</summary>
        [PexMethod(MaxRunsWithoutNewTests = 200)]
        public LevenshteinMatches LevenshteinSingleMatrixCPUTest(
            string str,
            string expression,
            int maxDistance,
            bool onlyBestResults,
            bool caseSensitive
        ) {
            LevenshteinMatches result = StringExtension.LevenshteinSingleMatrixCPU
                                            (str, expression, maxDistance, onlyBestResults, caseSensitive);
            return result;
            // TODO: add assertions to method StringExtensionTest.LevenshteinSingleMatrixCPUTest(String, String, Int32, Boolean, Boolean)
        }
    }
}
