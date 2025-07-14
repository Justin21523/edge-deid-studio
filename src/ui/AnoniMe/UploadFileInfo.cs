using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;

namespace AnoniMe
{
    public class UploadFileInfo : INotifyPropertyChanged
    {
        private string _fileName = string.Empty;
        private double _uploadProgress;
        private string _uploadStatus = string.Empty;

        public string FileName
        {
            get => _fileName;
            set
            {
                _fileName = value;
                OnPropertyChanged(nameof(FileName));
            }
        }

        public double UploadProgress
        {
            get => _uploadProgress;
            set
            {
                _uploadProgress = value;
                OnPropertyChanged(nameof(UploadProgress));
            }
        }

        public string UploadStatus
        {
            get => _uploadStatus;
            set
            {
                _uploadStatus = value;
                OnPropertyChanged(nameof(UploadStatus));
            }
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        private void OnPropertyChanged(string propertyName) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

}
