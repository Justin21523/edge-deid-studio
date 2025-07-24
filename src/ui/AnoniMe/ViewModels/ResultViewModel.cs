using AnoniMe.Models;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.UI.Xaml.Controls;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.Storage;

namespace AnoniMe.ViewModels
{

    public partial class ResultViewModel : ObservableObject
    {
        public ObservableCollection<PreviewFileInfo> PreviewFiles { get; } = new();

        public ObservableCollection<string> ProcessedFileNames { get; } = new();

        [RelayCommand]
        public void Download()
        {
            // TODO: 實作下載邏輯（可遍歷 PreviewFiles）
        }

        [RelayCommand]
        public void GoHome()
        {
            if (App.MainAppWindow?.Content is Frame rootFrame)
                rootFrame.Navigate(typeof(HomePage));
        }

        public async Task LoadFilesAsync(IEnumerable<StorageFile> files)
        {
            PreviewFiles.Clear();

            foreach (var file in files)
            {
                var props = await file.GetBasicPropertiesAsync();
                var info = new PreviewFileInfo
                {
                    FileName = file.Name,
                    FileType = file.FileType,
                    FileSizeDisplay = $"{props.Size / 1024.0 / 1024.0:F2} MB",
                    IsPdf = file.FileType.Equals(".pdf", StringComparison.OrdinalIgnoreCase),
                    IsDoc = file.FileType.Equals(".doc", StringComparison.OrdinalIgnoreCase) ||
                            file.FileType.Equals(".docx", StringComparison.OrdinalIgnoreCase)
                };

                if (info.IsPdf)
                {
                    info.PdfUri = $"file:///{file.Path}";
                    info.PreviewText = $"PDF 預覽：{file.Name}";
                }
                else if (info.IsDoc)
                {
                    using var stream = await file.OpenStreamForReadAsync();
                    using var reader = new StreamReader(stream, Encoding.UTF8);
                    info.PreviewText = await reader.ReadToEndAsync();
                }
                else
                {
                    info.PreviewText = "尚未支援此格式";
                }

                PreviewFiles.Add(info);
            }
        }
    }

}