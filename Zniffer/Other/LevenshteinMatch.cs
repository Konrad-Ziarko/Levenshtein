using CustomExtensions;

namespace Zniffer.Levenshtein {
    public class LevenshteinMatch {
        public string context { get; }
        public int distance { get; }
        public int position { get; }
        public int length { get; }

        public LevenshteinMatch(string context, int position, int length, int distance, int paddingLength = 10) {
            this.length = length;
            this.context = context.GetContext(ref position, length, paddingLength);
            this.position = position;
            this.distance = distance;
        }
    }
}
