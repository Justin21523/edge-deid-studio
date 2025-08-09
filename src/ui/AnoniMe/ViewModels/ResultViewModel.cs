using AnoniMe.Models;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.UI.Xaml.Controls;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Windows.Storage;

namespace AnoniMe.ViewModels
{
    /// <summary>
    /// 結果頁面的 ViewModel，負責處理檔案預覽與下載等功能。
    /// </summary>
    public partial class ResultViewModel : ObservableObject
    {
        // 儲存處理後的檔案預覽資訊（包括檔名、類型、大小、內容預覽等）。
        public ObservableCollection<PreviewFileInfo> PreviewFiles { get; } = new();

        // 已處理檔案名稱清單，可用於後續判斷或顯示。
        public ObservableCollection<string> ProcessedFileNames { get; } = new();

        // 下載檔案的指令（目前尚未實作，將遍歷 PreviewFiles）。
        [RelayCommand]
        public void Download()
        {
            // TODO: 實作下載邏輯（可遍歷 PreviewFiles）
        }


        // 返回首頁的指令
        [RelayCommand]
        public void GoHome()
        {
            if (App.MainAppWindow?.Content is Frame rootFrame)
                rootFrame.Navigate(typeof(HomePage));
        }


        // 載入並解析檔案，將結果加入 PreviewFiles 供 UI 顯示。
        public async Task LoadFilesAsync(IEnumerable<StorageFile> files)
        {
            PreviewFiles.Clear();

            foreach (var file in files)
            {
                // 取得檔案屬性（例如檔案大小）
                var props = await file.GetBasicPropertiesAsync();
                var info = new PreviewFileInfo
                {
                    FileName = file.Name,
                    FileType = file.FileType,
                    FileSizeDisplay = $"{props.Size / 1024.0 / 1024.0:F2} MB", // 顯示成 MB
                    IsPdf = file.FileType.Equals(".pdf", StringComparison.OrdinalIgnoreCase),
                    IsDoc = file.FileType.Equals(".doc", StringComparison.OrdinalIgnoreCase) ||
                            file.FileType.Equals(".docx", StringComparison.OrdinalIgnoreCase)
                };

                // 根據檔案格式處理預覽內容
                if (info.IsPdf)
                {
                    info.PdfUri = $"file:///{file.Path}";
                    info.PreviewText = $"PDF 預覽：{file.Name}";
                }
                // 讀取 Word 文件的文字內容（注意：這裡僅適用純文字，無法解析 DOCX 格式的複雜結構）
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

                // 將檔案加入預覽清單
                PreviewFiles.Add(info);
            }
        }
    }

}