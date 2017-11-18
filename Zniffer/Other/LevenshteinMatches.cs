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
            get; private set;
        }

        public LevenshteinMatches() {
            Lenght = 0;
            foundMatches = new List<LevenshteinMatch>();
        }
        public LevenshteinMatches(LevenshteinMatch match) {
            foundMatches = new List<LevenshteinMatch>();
            foundMatches.Add(match);
            Lenght = 1;
        }
        public LevenshteinMatches(List<LevenshteinMatch> matches) {
            foundMatches = matches;
            Lenght = foundMatches.Count;
        }

        public void addMatch(string match, int position, int len, int dist) {
            if (foundMatches == null)
                foundMatches = new List<LevenshteinMatch>();
            var newMatch = new LevenshteinMatch(match, position, len, dist);

            foundMatches.Add(newMatch);
            Lenght++;
        }
        public void addMatch(LevenshteinMatch match) {
            if (foundMatches == null)
                foundMatches = new List<LevenshteinMatch>();

            foundMatches.Add(match);
            Lenght++;
        }
        public bool removeMatch(int indexOf) {
            try {
                foundMatches.RemoveAt(indexOf);
                Lenght--;

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
        public string match { get; }
        public int position { get; }
        public int length { get; }
        public int distance { get; }

        public LevenshteinMatch(string match, int position, int length, int distance) {
            this.match = match;
            this.position = position;
            this.length = length;
            this.distance = distance;
        }
    }
}
