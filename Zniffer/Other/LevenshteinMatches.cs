using System;
using System.Collections.Generic;

namespace Zniffer.Levenshtein {
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

    
}
