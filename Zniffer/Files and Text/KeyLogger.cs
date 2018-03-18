using System;
using System.Text;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using System.Globalization;
using System.Threading;
using CustomExtensions;
using Zniffer.Levenshtein;
using System.Diagnostics;
using System.Collections.Generic;

namespace Zniffer.FilesAndText {
    class KeyLogger {
        #region DllImports
        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern IntPtr SetWindowsHookEx(int idHook, LowLevelKeyboardProc lpfn, IntPtr hMod, uint dwThreadId);

        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool UnhookWindowsHookEx(IntPtr hhk);

        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern IntPtr CallNextHookEx(IntPtr hhk, int nCode,

        IntPtr wParam, IntPtr lParam);
        [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern IntPtr GetModuleHandle(string lpModuleName);

        #endregion

        public delegate void keyCaptured(string s);

        public static System.Timers.Timer timerResetString = new System.Timers.Timer(5000);//5sec reset time

        #region modifiers
        static bool shift = false;
        static bool control = false;
        static bool alt = false;
        #endregion

        #region Keylogger
        public static string keyBuffer = "";
        public static long cursorPosition = 0;
        #endregion

        static string decimalSeparator = new CultureInfo(Thread.CurrentThread.CurrentCulture.Name, false).NumberFormat.NumberDecimalSeparator;

        private static MainWindow window;
        private const int WH_KEYBOARD_LL = 13;
        private const int WM_KEYDOWN = 0x0100;
        private static LowLevelKeyboardProc _proc = HookCallback;
        private static IntPtr _hookID = IntPtr.Zero;
        private static IntPtr SetHook(LowLevelKeyboardProc proc) {
            using (Process curProcess = Process.GetCurrentProcess())
            using (ProcessModule curModule = curProcess.MainModule) {
                return SetWindowsHookEx(WH_KEYBOARD_LL, proc,
                    GetModuleHandle(curModule.ModuleName), 0);
            }
        }

        private delegate IntPtr LowLevelKeyboardProc(
            int nCode, IntPtr wParam, IntPtr lParam);

        private static IntPtr HookCallback(
            int nCode, IntPtr wParam, IntPtr lParam) {
            if (nCode >= 0 && wParam == (IntPtr)WM_KEYDOWN) {
                int vkCode = Marshal.ReadInt32(lParam);
                
                if(Keys.Control == Control.ModifierKeys) {
                    control = true;
                }
                if(Keys.Shift == Control.ModifierKeys) {
                    shift = true;
                }
                if(Keys.RMenu == Control.ModifierKeys) {
                    alt = true;
                }

                if (Enum.GetName(typeof(Keys), vkCode) == "Back")
                    KeyCapturedHandle("<Backspace>");
                else if (Enum.GetName(typeof(Keys), vkCode) == "Space")
                    KeyCapturedHandle(" ");
                else if (Enum.GetName(typeof(Keys), vkCode) == "Return")
                    KeyCapturedHandle("\n");
                else if (Enum.GetName(typeof(Keys), vkCode) == "Delete")
                    KeyCapturedHandle("<Del>");
                else if (Enum.GetName(typeof(Keys), vkCode) == "Home")
                    KeyCapturedHandle("<Home>");
                else if (Enum.GetName(typeof(Keys), vkCode) == "End")
                    KeyCapturedHandle("<End>");
                else if (Enum.GetName(typeof(Keys), vkCode) == "Tab")
                    KeyCapturedHandle("<Tab>");
                else if (Enum.GetName(typeof(Keys), vkCode) == "Left")
                    KeyCapturedHandle("<Left>");
                else if (Enum.GetName(typeof(Keys), vkCode) == "Right")
                    KeyCapturedHandle("<Right>");
                else {
                    KeysConverter kc = new KeysConverter();
                    string keyChar = kc.ConvertToString(vkCode);

                    if (string.Equals(keyChar, "decimal", StringComparison.OrdinalIgnoreCase))
                        keyChar = ",";

                    if (!shift) {
                        keyChar = keyChar.ToLower();


                        if (string.Equals(keyChar, "oemminus", StringComparison.OrdinalIgnoreCase))
                            keyChar = "-";
                        else if (string.Equals(keyChar, "oemplus", StringComparison.OrdinalIgnoreCase))
                            keyChar = "=";
                        else if (string.Equals(keyChar, "oemtilde", StringComparison.OrdinalIgnoreCase))
                            keyChar = "`";
                        else if (string.Equals(keyChar, "oemopenbrackets", StringComparison.OrdinalIgnoreCase))
                            keyChar = "[";
                        else if (string.Equals(keyChar, "oem6", StringComparison.OrdinalIgnoreCase))
                            keyChar = "]";
                        else if (string.Equals(keyChar, "oem5", StringComparison.OrdinalIgnoreCase))
                            keyChar = @"\";
                        else if (string.Equals(keyChar, "oem7", StringComparison.OrdinalIgnoreCase))
                            keyChar = "'";
                        else if (string.Equals(keyChar, "oem1", StringComparison.OrdinalIgnoreCase))
                            keyChar = ";";
                        else if (string.Equals(keyChar, "oemquestion", StringComparison.OrdinalIgnoreCase))
                            keyChar = "/";
                        else if (string.Equals(keyChar, "oemperiod", StringComparison.OrdinalIgnoreCase))
                            keyChar = ".";
                        else if (string.Equals(keyChar, "oemcomma", StringComparison.OrdinalIgnoreCase))
                            keyChar = ",";
                        else if (string.Equals(keyChar, "decimal", StringComparison.OrdinalIgnoreCase))
                            keyChar = decimalSeparator;

                        else if (string.Equals(keyChar, "numpad1", StringComparison.OrdinalIgnoreCase))
                            keyChar = "1";
                        else if (string.Equals(keyChar, "numpad2", StringComparison.OrdinalIgnoreCase))
                            keyChar = "2";
                        else if (string.Equals(keyChar, "numpad3", StringComparison.OrdinalIgnoreCase))
                            keyChar = "3";
                        else if (string.Equals(keyChar, "numpad4", StringComparison.OrdinalIgnoreCase))
                            keyChar = "4";
                        else if (string.Equals(keyChar, "numpad5", StringComparison.OrdinalIgnoreCase))
                            keyChar = "5";
                        else if (string.Equals(keyChar, "numpad6", StringComparison.OrdinalIgnoreCase))
                            keyChar = "6";
                        else if (string.Equals(keyChar, "numpad7", StringComparison.OrdinalIgnoreCase))
                            keyChar = "7";
                        else if (string.Equals(keyChar, "numpad8", StringComparison.OrdinalIgnoreCase))
                            keyChar = "8";
                        else if (string.Equals(keyChar, "numpad9", StringComparison.OrdinalIgnoreCase))
                            keyChar = "9";
                        else if (string.Equals(keyChar, "numpad0", StringComparison.OrdinalIgnoreCase))
                            keyChar = "0";



                    }
                    else {
                        keyChar = keyChar.ToUpper();

                        if (string.Equals(keyChar, "oemminus", StringComparison.OrdinalIgnoreCase))
                            keyChar = "_";
                        else if (string.Equals(keyChar, "oemplus", StringComparison.OrdinalIgnoreCase))
                            keyChar = "=";
                        else if (string.Equals(keyChar, "oemtilde", StringComparison.OrdinalIgnoreCase))
                            keyChar = "~";
                        else if (string.Equals(keyChar, "oemopenbrackets", StringComparison.OrdinalIgnoreCase))
                            keyChar = "{";
                        else if (string.Equals(keyChar, "oem6", StringComparison.OrdinalIgnoreCase))
                            keyChar = "}";
                        else if (string.Equals(keyChar, "oem5", StringComparison.OrdinalIgnoreCase))
                            keyChar = "|";
                        else if (string.Equals(keyChar, "oem7", StringComparison.OrdinalIgnoreCase))
                            keyChar = "\"";
                        else if (string.Equals(keyChar, "oem1", StringComparison.OrdinalIgnoreCase))
                            keyChar = ":";
                        else if (string.Equals(keyChar, "oemquestion", StringComparison.OrdinalIgnoreCase))
                            keyChar = "?";
                        else if (string.Equals(keyChar, "oemperiod", StringComparison.OrdinalIgnoreCase))
                            keyChar = ">";
                        else if (string.Equals(keyChar, "oemcomma", StringComparison.OrdinalIgnoreCase))
                            keyChar = "<";
                        else if (string.Equals(keyChar, "decimal", StringComparison.OrdinalIgnoreCase))
                            keyChar = "<Del>";
                        else if (string.Equals(keyChar, "1", StringComparison.OrdinalIgnoreCase))
                            keyChar = "!";
                        else if (string.Equals(keyChar, "2", StringComparison.OrdinalIgnoreCase))
                            keyChar = "@";
                        else if (string.Equals(keyChar, "3", StringComparison.OrdinalIgnoreCase))
                            keyChar = "#";
                        else if (string.Equals(keyChar, "4", StringComparison.OrdinalIgnoreCase))
                            keyChar = "$";
                        else if (string.Equals(keyChar, "5", StringComparison.OrdinalIgnoreCase))
                            keyChar = "%";
                        else if (string.Equals(keyChar, "6", StringComparison.OrdinalIgnoreCase))
                            keyChar = "^";
                        else if (string.Equals(keyChar, "7", StringComparison.OrdinalIgnoreCase))
                            keyChar = "&";
                        else if (string.Equals(keyChar, "8", StringComparison.OrdinalIgnoreCase))
                            keyChar = "*";
                        else if (string.Equals(keyChar, "9", StringComparison.OrdinalIgnoreCase))
                            keyChar = "(";
                        else if (string.Equals(keyChar, "0", StringComparison.OrdinalIgnoreCase))
                            keyChar = ")";

                    }

                    if (string.Equals(keyChar, "divide", StringComparison.OrdinalIgnoreCase))
                        keyChar = "/";
                    else if (string.Equals(keyChar, "multiply", StringComparison.OrdinalIgnoreCase))
                        keyChar = "*";
                    else if (string.Equals(keyChar, "subtract", StringComparison.OrdinalIgnoreCase))
                        keyChar = "-";
                    else if (string.Equals(keyChar, "add", StringComparison.OrdinalIgnoreCase))
                        keyChar = "+";
                    else if (string.Equals(keyChar, "enter", StringComparison.OrdinalIgnoreCase))
                        keyChar = "\n";

                    if (alt) {
                        if (string.Equals(keyChar, "a", StringComparison.OrdinalIgnoreCase))
                            keyChar = "ą";
                        else if (string.Equals(keyChar, "z", StringComparison.OrdinalIgnoreCase))
                            keyChar = "ż";
                        else if (string.Equals(keyChar, "x", StringComparison.OrdinalIgnoreCase))
                            keyChar = "ź";
                        else if (string.Equals(keyChar, "c", StringComparison.OrdinalIgnoreCase))
                            keyChar = "ć";
                        else if (string.Equals(keyChar, "e", StringComparison.OrdinalIgnoreCase))
                            keyChar = "ę";
                        else if (string.Equals(keyChar, "s", StringComparison.OrdinalIgnoreCase))
                            keyChar = "ś";
                        else if (string.Equals(keyChar, "n", StringComparison.OrdinalIgnoreCase))
                            keyChar = "ń";
                        else if (string.Equals(keyChar, "o", StringComparison.OrdinalIgnoreCase))
                            keyChar = "ó";
                        else if (string.Equals(keyChar, "l", StringComparison.OrdinalIgnoreCase))
                            keyChar = "ł";

                    }

                    KeyCapturedHandle(keyChar);
                }
                control = shift = alt = false;
            }
            return CallNextHookEx(_hookID, nCode, wParam, lParam);
        }



        public KeyLogger(MainWindow window) {
            KeyLogger.window = window;

            _hookID = SetHook(_proc);
        }
        ~KeyLogger() {
            UnhookWindowsHookEx(_hookID);
        }

        public void OnResetTimerEvent(object source, System.Timers.ElapsedEventArgs e) {
            cursorPosition = 0;
            keyBuffer = "";
        }

        private static void KeyCapturedHandle(string s) {
            if (s.Substring(0, 1).Equals("[") && s.Substring(s.Length - 1, 1).Equals("]")) {

            }
            else {

                if (s.Substring(0, 1).Equals("<") && s.Substring(s.Length - 1, 1).Equals(">")) {//special characters
                    s = s.Substring(1, s.Length - 2);
                    if (s.Equals("Backspace")) {
                        timerResetString.Stop();
                        if (keyBuffer.Length > 0) {
                            keyBuffer = keyBuffer.Remove(keyBuffer.Length - 1);
                            cursorPosition--;
                        }
                    }
                    else if (s.Equals("Left")) {
                        timerResetString.Stop();

                        if (cursorPosition > 0)
                            cursorPosition--;
                    }
                    else if (s.Equals("Right")) {
                        timerResetString.Stop();

                        if (cursorPosition < keyBuffer.Length)
                            cursorPosition++;
                    }
                }
                else {//normal characters
                    timerResetString.Stop();

                    keyBuffer += s;
                    cursorPosition++;

                    if (s.Equals("\n")) {//user returned(ended) string
                        cursorPosition = 0;
                        keyBuffer = "";
                    }
                }

                if (keyBuffer.Length > 50)
                    keyBuffer = keyBuffer.Remove(0, 1);
                Console.Out.WriteLine(keyBuffer);
                string phrase = MainWindow.SearchPhrase;
                LevenshteinMatches result = keyBuffer.Levenshtein(phrase, mode: MainWindow.SearchMode);

                if (result != null && result.hasMatches) {
                    window.AddTextToClipBoardBox(result);
                }

                timerResetString.Start();
            }
        }
    }
}
