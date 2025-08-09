using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AnoniMe.Models
{
    public partial class UploadFileInfo : ObservableObject
    {
        [ObservableProperty] private string fileName = string.Empty;
        [ObservableProperty] private string? fullPath;      // 重要：傳給 Python 用
        [ObservableProperty] private int uploadProgress;
        [ObservableProperty] private string uploadStatus = string.Empty;
        [ObservableProperty] private string? outputPath;    // Python 輸出檔路徑
    }
}
