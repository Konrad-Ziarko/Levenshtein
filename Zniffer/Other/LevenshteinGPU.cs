using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Collections.Generic;
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
                int NUMBER_OF_BLOCKS = source.Length + 2 - pattern.Length;
                byte compareLength = (byte)pattern.Length;

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
                

                lock (lockGPU) {

                    byte[,,] dev_levMatrix = _gpu.Allocate<byte>(NUMBER_OF_BLOCKS, compareLength + 1, compareLength + 1);
                    char[] host_source = source.ToCharArray();
                    char[] host_pattern = pattern.ToCharArray();
                    byte[] host_results = new byte[NUMBER_OF_BLOCKS];
                    byte[] dev_results = _gpu.Allocate<byte>(NUMBER_OF_BLOCKS);
                    char[] dev_source = _gpu.CopyToDevice(host_source);
                    char[] dev_pattern = _gpu.CopyToDevice(host_pattern);


                    //launch kernel
                    //_gpu.Launch(firstDim, 1).LevenshteinGpu(dev_source, dev_pattern, dev_levMatrix, firstDim, compareLength, dev_results);

                    int THREADS_PER_BLOCK = 512;

                    if (NUMBER_OF_BLOCKS >= THREADS_PER_BLOCK)
                        _gpu.Launch(NUMBER_OF_BLOCKS / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1).LevenshteinGpu3(dev_source, dev_pattern, dev_levMatrix, NUMBER_OF_BLOCKS, compareLength, dev_results);

                    //_gpu.Launch(NUMBER_OF_BLOCKS / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1).LevenshteinGpu3(dev_source, dev_pattern, dev_levMatrix, NUMBER_OF_BLOCKS, compareLength, dev_results);
                    else
                        _gpu.Launch(NUMBER_OF_BLOCKS, 1).LevenshteinGpu(dev_source, dev_pattern, dev_levMatrix, NUMBER_OF_BLOCKS, compareLength, dev_results);

                    _gpu.CopyFromDevice(dev_results, host_results);


                    for (int i = 0; i < NUMBER_OF_BLOCKS; i++) {
                        if (maxDistance >= host_results[i]) {
                            newMatches.Add(new LevenshteinMatch(originalString, i, compareLength, host_results[i]));
                        }
                    }
                    _gpu.Free(dev_results);
                    _gpu.Free(dev_source);
                    _gpu.Free(dev_pattern);
                    _gpu.HostFreeAll();

                    host_source = host_pattern = dev_source = dev_pattern = null;
                    host_results = null;
                }
                return new LevenshteinMatches(newMatches);
            }
            finally {
                if (_gpu != null) {
                    
                    _gpu.FreeAll();
                    GC.Collect();

                }
            }
        }

        [Cudafy]
        private static void LevenshteinGpu3(GThread thread, char[] source, char[] pattern, byte[,,] levMatrix, int firstDim, byte compareLength, byte[] dev_results) {
            int tid = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            for (byte j = 0; j <= compareLength; j++) {
                levMatrix[tid, 0, j] = j;
                levMatrix[tid, j, 0] = j;
            }

            if (tid < firstDim) {
                for (int i = 1; i <= compareLength; i++) {
                    for (int j = 1; j <= compareLength; j++) {
                        int iMinusOne = i - 1;
                        int jMinusOne = j - 1;

                        if (tid + iMinusOne < source.Length && source[tid + iMinusOne] == pattern[jMinusOne]) {
                            levMatrix[tid, i, j] = levMatrix[tid, iMinusOne, jMinusOne];
                        }
                        else {
                            byte x = levMatrix[tid, iMinusOne, j];
                            if (x > levMatrix[tid, i, jMinusOne])
                                x = levMatrix[tid, i, jMinusOne];
                            if (x > levMatrix[tid, iMinusOne, jMinusOne])
                                x = levMatrix[tid, iMinusOne, jMinusOne];
                            levMatrix[tid, i, j] = ++x;
                        }
                    }
                }
                dev_results[tid] = levMatrix[tid, compareLength, compareLength];
            }
        }

        [Cudafy]
        private static void LevenshteinGpu2(GThread thread, char[] source, char[] pattern, byte[,,] levMatrix, int firstDim, byte compareLength, byte[] dev_results) {
            int tid = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            for (byte j = 0; j <= compareLength; j++) {
                levMatrix[tid, 0, j] = j;
                levMatrix[tid, j, 0] = j;
            }

            if (tid < firstDim) {
                for (int i = 1; i <= compareLength; i++) {
                    for (int j = 1; j <= compareLength; j++) {
                        if (tid + i - 1 < source.Length && source[tid + i - 1] == pattern[j - 1]) {
                            levMatrix[tid, i, j] = levMatrix[tid, i - 1, j - 1];
                        }
                        else {
                            levMatrix[tid, i, j] = (byte)(Math.Min(Math.Min(levMatrix[tid, i - 1, j], levMatrix[tid, i, j - 1]), levMatrix[tid, i - 1, j - 1]) + 1);
                        }
                    }
                }
                dev_results[tid] = levMatrix[tid, compareLength, compareLength];
            }
        }

        [Cudafy]
        private static void LevenshteinGpu(GThread thread, char[] source, char[] pattern, byte[,,] levMatrix, int firstDim, byte compareLength, byte[] dev_results) {
            int tid = thread.blockIdx.x;

            for (byte j = 0; j <= compareLength; j++) {
                levMatrix[tid, 0, j] = j;
                levMatrix[tid, j, 0] = j;
            }

            if (tid < firstDim) {
                for (int i = 1; i <= compareLength; i++) {
                    for (int j = 1; j <= compareLength; j++) {
                        if (tid + i - 1 < source.Length && source[tid + i - 1] == pattern[j - 1]) {
                            levMatrix[tid, i, j] = levMatrix[tid, i - 1, j - 1];
                        }
                        else {
                            levMatrix[tid, i, j] = (byte)(Math.Min(Math.Min(levMatrix[tid, i - 1, j], levMatrix[tid, i, j - 1]), levMatrix[tid, i - 1, j - 1]) + 1);
                        }
                    }
                }
                dev_results[tid] = levMatrix[tid, compareLength, compareLength];
            }
        }
    }
}
