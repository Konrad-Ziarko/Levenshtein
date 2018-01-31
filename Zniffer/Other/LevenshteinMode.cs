using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zniffer.Levenshtein {
    public enum LevenshteinMode {
        SingleMatixCPU = 0,
        SplitForSingleMatrixCPU,
        MultiMatrixSingleThread,
        MultiMatrixParallel,
        DualRowCPU,
    }
}
