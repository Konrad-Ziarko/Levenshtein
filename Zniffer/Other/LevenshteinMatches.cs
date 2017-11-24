using CustomExtensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zniffer {
    public class LevenshteinMatches {

        public bool hasMatches {
            get { return Lenght > 0; }
        }
        public List<LevenshteinMatch> foundMatches = null;
        public int Lenght {
            get { return foundMatches.Count; }
        }

        public LevenshteinMatches() {
            foundMatches = new List<LevenshteinMatch>();
        }
        public LevenshteinMatches(LevenshteinMatch match) {
            foundMatches = new List<LevenshteinMatch>();
            foundMatches.Add(match);
        }
        public LevenshteinMatches(List<LevenshteinMatch> matches) {
            foundMatches = matches;
        }
        public LevenshteinMatches(LevenshteinMatches a, LevenshteinMatches b) {
            //maybe should check if a/b are null ref
            foundMatches = new List<LevenshteinMatch>(a.Lenght + b.Lenght);
            foundMatches.AddRange(a.foundMatches);
            foundMatches.AddRange(b.foundMatches);
        }

        public void addMatch(string context, int position, int len, int dist) {
            if (foundMatches == null)
                foundMatches = new List<LevenshteinMatch>();
            var newMatch = new LevenshteinMatch(context, position, len, dist);

            foundMatches.Add(newMatch);
        }
        public void addMatch(LevenshteinMatch match) {
            if (foundMatches == null)
                foundMatches = new List<LevenshteinMatch>();

            foundMatches.Add(match);
        }
        public bool removeMatch(int indexOf) {
            try {
                foundMatches.RemoveAt(indexOf);

                return true;
            }
            catch (ArgumentOutOfRangeException) {
                return false;
            }
        }

        public void removeMatches(int minDistance) {
            for (int i = foundMatches.Count - 1; i >= 0; i--) {
                if (foundMatches[i].distance > minDistance)
                    foundMatches.RemoveAt(i);
            }
        }
    }

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
