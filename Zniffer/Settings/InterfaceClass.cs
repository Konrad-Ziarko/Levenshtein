using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;
using System.Windows.Media;

namespace Zniffer {
    [Serializable()]
    public class InterfaceClass : INotifyPropertyChanged {
        private string _ports;
        private string _addres;
        private bool _used;
        private bool _interfaceIsUp;
        private bool _continueCapturing;
        public byte[] byteData = new byte[4096];

        public string Addres {
            get; set;
        }
        public string Ports {
            get {
                return _ports;
            }
            set {
                if (_ports != value) {
                    _ports = value;
                    OnPropertyChanged("ports");
                }
            }
        }
        public bool Used {
            get; set;
        }
        public bool InterfaceIsUp {
            get {
                return _interfaceIsUp;
            }
            set {
                if (_interfaceIsUp != value) {
                    _interfaceIsUp = value;
                    OnPropertyChanged("statusUp");
                }
            }
        }

        public bool ContinueCapturing {
            get {
                return _continueCapturing;
            }
            set {
                _continueCapturing = value;
            }
        }

        public InterfaceClass(string addres, string ports) {
            this.Addres = addres;
            this.Ports = ports;
            this.Used = false;
            this.InterfaceIsUp = false;
            this.ContinueCapturing = true;
        }
        public InterfaceClass(string addres, string ports, bool used) {
            this.Addres = addres;
            this.Ports = ports;
            this.Used = used;
            this.InterfaceIsUp = false;
            this.ContinueCapturing = true;
        }
        private void OnPropertyChanged(string propertyName) {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public event PropertyChangedEventHandler PropertyChanged;

        public override string ToString() {
            return Addres + ":" + Ports;
        }
    }
}
