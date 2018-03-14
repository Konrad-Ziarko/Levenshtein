using System;
using System.ComponentModel;

namespace Zniffer {
    [Serializable()]
    public class InterfaceClass : INotifyPropertyChanged {
        private string _ports;
        private bool _interfaceIsUp;
        private ushort _minPort=0, _maxPort=0;

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
                    try {
                        if (this.Ports.Contains("-")) {
                            string[] minMax = this.Ports.Split('-');
                            _minPort = Convert.ToUInt16(minMax[0]);
                            _maxPort = Convert.ToUInt16(minMax[1]);
                        }
                        else
                            _minPort = _maxPort = Convert.ToUInt16(this.Ports);
                    }
                    catch {
                        _minPort = _maxPort = 0;
                    }
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
            this.InterfaceIsUp = false;
        }
        public InterfaceClass(string addres, string ports, bool used) {
            this.Addres = addres;
            this.Ports = ports;
            this.Used = used;
            this.InterfaceIsUp = false;
        }
        private void OnPropertyChanged(string propertyName) {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public event PropertyChangedEventHandler PropertyChanged;

        public override string ToString() {
            return Addres + ":" + Ports;
        }

        public bool isPortValid(ushort portNo) {
            if ((_minPort == 0 && _maxPort == 0) || (portNo >= _minPort && portNo <= _maxPort))
                return true;
            return false;
        }
    }
}
