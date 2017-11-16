
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Zniffer;

namespace CustomExtensions {
    public static class StringExtension {

        #region CUDA
        /*
        static void SquareKernel(deviceptr outputs, deviceptr inputs, int n) {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < n; i += stride) {
                outputs[i] = inputs[i] * inputs[i];
            }
        }

        static double[] SquareGPU(double[] inputs) {
            var worker = Worker.Default;
            using (var dInputs = worker.Malloc(inputs))
            using (var dOutputs = worker.Malloc(inputs.Length)) {
                const int blockSize = 256;
                var numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT;
                var gridSize = Math.Min(16 * numSm,
                                        Common.divup(inputs.Length, blockSize));
                var lp = new LaunchParam(gridSize, blockSize);
                worker.Launch(SquareKernel, lp, dOutputs.Ptr, dInputs.Ptr,
                              inputs.Length);
                return dOutputs.Gather();
            }
        }*/

        #endregion


        public static LevenshteinMatch LevenshteinCPU(this string str, string expression, int maxDistance) {
            string source = str;
            source = source.ToUpper();
            expression = expression.ToUpper();

            int[,] dimension = new int[source.Length + 1, expression.Length + 1];
            int matchCost = 0;
            if (source.Length == 0 || expression.Length == 0) {
                return null;
            }

            if (source.Length == expression.Length)
                for (int i = 0; i <= source.Length; i++) {
                    dimension[i, 0] = i;
                    dimension[0, i] = i;
                }
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

            double percentage = Math.Round((1.0 - ((double)dimension[source.Length, expression.Length] / (double)Math.Max(source.Length, expression.Length))) * 100.0, 2);

            if (dimension[source.Length, expression.Length] <= maxDistance)
                return new LevenshteinMatch(str, percentage, 0, source.Length, dimension[source.Length, expression.Length]);
            else
                return null;
        }
    }
}
