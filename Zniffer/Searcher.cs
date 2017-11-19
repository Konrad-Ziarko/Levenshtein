using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CustomExtensions;


namespace Zniffer {
    class Searcher {


        public static List<string> GetDirectories(string path, string searchPattern = "*",
        SearchOption searchOption = SearchOption.TopDirectoryOnly) {
            if (searchOption == SearchOption.TopDirectoryOnly)
                return Directory.GetDirectories(path, searchPattern).ToList();

            var directories = new List<string>(GetDirectories(path, searchPattern));

            for (var i = 0; i < directories.Count; i++)
                directories.AddRange(GetDirectories(directories[i], searchPattern));

            return directories;
        }

        private static List<string> GetDirectories(string path, string searchPattern) {
            try {
                return Directory.GetDirectories(path, searchPattern).ToList();
            }
            catch (UnauthorizedAccessException) {
                return new List<string>();
            }
        }

        public static List<string> GetFiles(string path) {
            string searchPattern = "*";
            try {
                return Directory.GetFiles(path, searchPattern).ToList();
            }
            catch (UnauthorizedAccessException) {
                return new List<string>();
            }
        }

        public static LevenshteinMatches ExtractPhrase(string sourceText) {
            //StringBuilder sb = new StringBuilder();
            string phrase = MainWindow.searchPhrase;
            LevenshteinMatches matches = sourceText.LevenshteinMultiMatrixParallel(phrase);

            return matches;
        }

        public static async Task<string> ReadTextAsync(string filePath) {
            StringBuilder sb = new StringBuilder();
            string textFromFile = File.ReadAllText(filePath);
            sb.Append(ExtractPhrase(textFromFile));
            return sb.ToString();
        }
        /*
        public static List<int> AllIndexesOf(string str, string value) {
            if (String.IsNullOrEmpty(value))
                throw new ArgumentException("the string to find can not be empty", "value");
            List<int> indexes = new List<int>();
            for (int index = 0; ; index += value.Length) {
                index = str.IndexOf(value, index);
                if (index == -1)
                    return indexes;
                indexes.Add(index);
            }
        }*/
    }
}
