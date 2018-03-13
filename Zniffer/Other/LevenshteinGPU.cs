using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace Zniffer.Other {
    public class LevenshteinGPU {

        public void Run() {
            CudafyModule km = CudafyTranslator.Cudafy();
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

            //alokowac tablice
            //wysłać na karte

            //wywolac kernel

            //odebrac wynik z gpu


            gpu.FreeAll();

        }
        


        [Cudafy]
        private static void LevenshteinGpu(string s, GThread thread, int[] A, int[] B, int[,] AB) {

            
        }
    }
}
