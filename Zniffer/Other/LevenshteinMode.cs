
namespace Zniffer {
    public enum LevenshteinMode {
        SingleMatixCPU = 0,
        SplitForSingleMatrixCPU,
        MultiMatrixSingleThreadCPU,
        MultiMatrixParallelCPU,
        SplitDualRowCPU,
        ThreeDimMatrixCPU,
        ThreeDimMatrixGPU,
        ThreeDimMatrixParallelCPU,
        SplitForParallelCPU,
        HistogramCPU,
    }
}
