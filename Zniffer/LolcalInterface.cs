using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zniffer
{
    [Serializable()]
    public class InterfaceClass : INotifyPropertyChanged
    {
        public string _ports;
        public string _addres;
        public bool _used;

        public string addres {
            get;set;
        }
        public string ports {
            get {
                return _ports;
            }
            set {
                if (_ports != value)
                {
                    _ports = value;
                    OnPropertyChanged("ports");
                }
            }
        }
        public bool used {
            get;set;
        }

        public InterfaceClass(string addres, string ports)
        {
            this.addres = addres;
            this.ports = ports;
            this.used = false;
        }
        public InterfaceClass(string addres, string ports, bool used)
        {
            this.addres = addres;
            this.ports = ports;
            this.used = used;
        }
        private void OnPropertyChanged(string propertyName)
        {
            PropertyChangedEventHandler handler = PropertyChanged;
            if (handler != null)
            {
                handler(this, new PropertyChangedEventArgs(propertyName));
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        public override string ToString()
        {
            return addres + ":" + ports;
             
        }
    }
}
