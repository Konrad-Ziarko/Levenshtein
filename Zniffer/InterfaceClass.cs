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
        public string _ports;
        public string _addres;
        public bool _used;
        public bool _interfaceIsUp;

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

        public InterfaceClass(string addres, string ports) {
            this.Addres = addres;
            this.Ports = ports;
            this.Used = false;
            InterfaceIsUp = false;
        }
        public InterfaceClass(string addres, string ports, bool used) {
            this.Addres = addres;
            this.Ports = ports;
            this.Used = used;
            InterfaceIsUp = false;
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
