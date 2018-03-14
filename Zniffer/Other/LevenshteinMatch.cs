using CustomExtensions;

namespace Zniffer.Levenshtein {
    public class LevenshteinMatch {
        public string contextL { get; }
        public string contextR { get; }
        public string value { get; }
        public int distance { get; }
        public int position { get; }
        public int length { get; }

        public LevenshteinMatch(string context, int position, int length, int distance, int paddingLength = 20, string expression = null) {
            int startPosition = position - paddingLength;
            if (startPosition < 0)
                startPosition = 0;
            int endPosiiton = position + length + paddingLength;
            if (endPosiiton > context.Length)
                endPosiiton = context.Length - (position + length);
            if (length >= context.Length || position >= context.Length) {
                this.value = context;
                this.contextL = this.contextR = "";
            }
            else if (position + length >= context.Length) {
                this.contextL = context.Substring(startPosition, position - startPosition);
                this.value = context.Substring(position);
                this.contextR = "";
            }
            else {
                this.contextL = context.Substring(startPosition, position - startPosition);
                this.value = context.Substring(position, length);
                this.contextR = context.Substring(position + length, endPosiiton - (position + length));
            }
            this.length = length;
            this.position = position;
            this.distance = distance;
        }
    }
}
