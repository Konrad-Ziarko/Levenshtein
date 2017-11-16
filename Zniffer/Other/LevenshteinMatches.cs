﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zniffer {
    public class LevenshteinMatches {
        
        public bool hasMatches = false;
        public List<LevenshteinMatch> foundMatches = null;

        public LevenshteinMatches() {
            foundMatches = new List<LevenshteinMatch>();
        }
        public LevenshteinMatches(LevenshteinMatch match) {
            foundMatches = new List<LevenshteinMatch>();
            foundMatches.Add(match);
            hasMatches = true;
        }
        public LevenshteinMatches(List<LevenshteinMatch> matches) {
            foundMatches = matches;
            hasMatches = true;
        }

        public void addMatch(string match, int position, int len, int dist) {
            if (foundMatches == null)
                foundMatches = new List<LevenshteinMatch>();
            var newMatch = new LevenshteinMatch(match, position, len, dist);

            foundMatches.Add(newMatch);
            hasMatches = true;
        }
        public void addMatch(LevenshteinMatch match) {
            if (foundMatches == null)
                foundMatches = new List<LevenshteinMatch>();

            foundMatches.Add(match);
            hasMatches = true;
        }
    }

    public class LevenshteinMatch {
        string match;
        int position;
        int len;
        int dist;

        public LevenshteinMatch(string match, int position, int len, int dist) {
            this.match = match;
            this.position = position;
            this.len = len;
            this.dist = dist;
        }
    }
}
