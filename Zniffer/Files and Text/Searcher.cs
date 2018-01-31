using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CustomExtensions;
using Zniffer.Properties;
using Zniffer.Levenshtein;

namespace Zniffer.FilesAndText {
    class Searcher {
        private MainWindow window;

        public Searcher(MainWindow window) {
            this.window = window;
        }

        public void SearchFiles(List<string> files, DriveInfo drive) {
            foreach (string file in files) {
                //Console.Out.WriteLine(File.ReadAllText(file));
                try {
                    LevenshteinMatches matches = ReadTextFromFile(file);
                    //foreach(string str in File.ReadLines(file))
                    if (matches.hasMatches) {
                        window.AddTextToFileBox(file);
                        window.AddTextToFileBox(matches);
                        window.AddTextToFileBox("");
                    }
                }
                catch (UnauthorizedAccessException) {
                    window.AddTextToFileBox("Cannot access:" + file);
                }
                catch (IOException) {
                    //odłączenie urządzenia np
                }
                catch (ArgumentException) {

                }
                if (Settings.Default.ScanADS && drive.DriveFormat.Equals("NTFS")) {
                    //search for ads
                }
            }
        }

        public List<string> GetDirectories(string path, string searchPattern = "*",
        SearchOption searchOption = SearchOption.TopDirectoryOnly) {
            if (searchOption == SearchOption.TopDirectoryOnly)
                return Directory.GetDirectories(path, searchPattern).ToList();

            var directories = new List<string>(GetDirectories(path, searchPattern));

            for (var i = 0; i < directories.Count; i++)
                directories.AddRange(GetDirectories(directories[i], searchPattern));

            return directories;
        }

        private List<string> GetDirectories(string path, string searchPattern) {
            try {
                return Directory.GetDirectories(path, searchPattern).ToList();
            }
            catch (UnauthorizedAccessException) {
                return new List<string>();
            }
        }

        public List<string> GetFiles(string path) {
            string searchPattern = "*";
            try {
                return Directory.GetFiles(path, searchPattern).ToList();
            }
            catch (UnauthorizedAccessException) {
                return new List<string>();
            }
        }

        public LevenshteinMatches ExtractPhrase(string sourceText) {
            //StringBuilder sb = new StringBuilder();
            string phrase = MainWindow.SearchPhrase;
            LevenshteinMatches matches = sourceText.Levenshtein(phrase, mode: LevenshteinMode.SplitForSingleMatrixCPU);

            return matches;
        }

        public LevenshteinMatches ReadTextFromFile(string filePath) {
            string textFromFile = File.ReadAllText(filePath);
            return ExtractPhrase(textFromFile);
        }
    }
}
