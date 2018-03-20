using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Collections.Generic;
using System.IO;
using Zniffer.Levenshtein;

namespace Zniffer.Other {
    public class LevenshteinGPU {

        private static GPGPU _gpu;
        private static LevenshteinGPU _instance;
        public static LevenshteinGPU instance {
            get {
                if (_instance == null) {
                    _instance = new LevenshteinGPU();
                }
                return _instance;
            }
        }
        private LevenshteinGPU() {
            CudafyModule km = null;
            try {
                km = CudafyModule.Deserialize(typeof(LevenshteinGPU).Name);
            }
            catch {
                km = CudafyTranslator.Cudafy(eArchitecture.sm_50);
            }
            _gpu = CudafyHost.GetDevice(CudafyModes.Target);
            _gpu.LoadModule(km);
        }

        private object lockGPU = new object();
        public LevenshteinMatches LevenshteinSingleMatrixGPU(string originalString, string pattern, int maxDistance = -1, bool onlyBestResults = false, bool caseSensitive = false) {
            List<LevenshteinMatch> newMatches = new List<LevenshteinMatch>();
            try {
                string source = originalString;
                int firstDim = source.Length + 2 - pattern.Length;
                int compareLength = pattern.Length;

                if (maxDistance < 0)
                    maxDistance = pattern.Length / 2;

                if (source.Length == 0 || pattern.Length == 0)
                    return new LevenshteinMatches();
                if (source.Length < pattern.Length)
                    return new LevenshteinMatches();

                if (!caseSensitive) {
                    source = source.ToUpper();
                    pattern = pattern.ToUpper();
                }

                int[,,] host_levMatrix = new int[firstDim, compareLength + 1, compareLength + 1];
                for (int i = 0; i < firstDim; i++) {
                    for (int j = 0; j <= compareLength; j++) {
                        host_levMatrix[i, 0, j] = j;
                        host_levMatrix[i, j, 0] = j;
                    }
                }
                lock (lockGPU) {
                    int[,,] dev_levMatrix = _gpu.CopyToDevice(host_levMatrix);

                    char[] host_source = source.ToCharArray();
                    char[] host_pattern = pattern.ToCharArray();
                    int[] host_results = new int[firstDim];
                    int[] dev_results = _gpu.Allocate<int>(firstDim);
                    char[] dev_source = _gpu.CopyToDevice(host_source);
                    char[] dev_pattern = _gpu.CopyToDevice(host_pattern);


                    //launch kernel
                    //_gpu.Launch(firstDim, 1).LevenshteinGpu(dev_source, dev_pattern, dev_levMatrix, firstDim, compareLength, dev_results);
                    _gpu.Launch(firstDim / 512, 512, 1).LevenshteinGpu2(dev_source, dev_pattern, dev_levMatrix, firstDim, compareLength, dev_results);

                    _gpu.CopyFromDevice(dev_results, host_results);


                    for (int i = 0; i < firstDim; i++) {
                        if (maxDistance >= host_results[i]) {
                            newMatches.Add(new LevenshteinMatch(originalString, i, compareLength, host_results[i]));
                        }
                    }
                }
                return new LevenshteinMatches(newMatches);
            }
            finally {
                if (_gpu != null)
                    _gpu.FreeAll();
            }
        }

        [Cudafy]
        private static void LevenshteinGpu2(GThread thread, char[] source, char[] pattern, int[,,] levMatrix, int firstDim, int compareLength, int[] dev_results) {
            int tid = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            //int tid = thread.blockIdx.x;

            //if(tid < firstDim) {
            for (int i = 1; i <= compareLength; i++) {
                for (int j = 1; j <= compareLength; j++) {
                    if (tid + i - 1 < source.Length && source[tid + i - 1] == pattern[j - 1]) {
                        levMatrix[tid, i, j] = levMatrix[tid, i - 1, j - 1];
                    }
                    else {
                        levMatrix[tid, i, j] = Math.Min(Math.Min(levMatrix[tid, i - 1, j], levMatrix[tid, i, j - 1]), levMatrix[tid, i - 1, j - 1]) + 1;
                    }
                }
            }
            dev_results[tid] = levMatrix[tid, compareLength, compareLength];
            //}
        }

        [Cudafy]
        private static void LevenshteinGpu(GThread thread, char[] source, char[] pattern, int[,,] levMatrix, int firstDim, int compareLength, int[] dev_results) {
            int tid = thread.blockIdx.x;

            //if(tid < firstDim) {
            for (int i = 1; i <= compareLength; i++) {
                for (int j = 1; j <= compareLength; j++) {
                    if (tid + i - 1 < source.Length && source[tid + i - 1] == pattern[j - 1]) {
                        levMatrix[tid, i, j] = levMatrix[tid, i - 1, j - 1];
                    }
                    else {
                        levMatrix[tid, i, j] = Math.Min(Math.Min(levMatrix[tid, i - 1, j], levMatrix[tid, i, j - 1]), levMatrix[tid, i - 1, j - 1]) + 1;
                    }
                }
            }
            dev_results[tid] = levMatrix[tid, compareLength, compareLength];
            //}
        }
    }
}
