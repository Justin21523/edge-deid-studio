using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnoniMe.Models
{
    public class UploadFileInfo
    {
        public string FileName { get; set; } = string.Empty;
        public int UploadProgress { get; set; }
        public string UploadStatus { get; set; } = string.Empty;
    }
}