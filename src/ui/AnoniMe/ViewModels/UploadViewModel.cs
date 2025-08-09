using AnoniMe.Models;
using AnoniMe.Services;
using AnoniMe.Views;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Windows.ApplicationModel.DataTransfer;
using Windows.Storage;
using Windows.Storage.Pickers;

namespace AnoniMe.ViewModels
{
    /// <summary>
    /// 檔案上傳頁面的 ViewModel，負責管理檔案選取、拖曳、標籤與上傳流程。
    /// 使用 MVVM Toolkit 提供的 ObservableObject 與 RelayCommand。
    /// </summary>
    /// 
    public partial class UploadViewModel : ObservableObject
    {
        // 新增欄位：透過 DI 注入
        private readonly PythonProcessor _python;

        // 儲存目前上傳檔案。
        public ObservableCollection<UploadFileInfo> UploadItems { get; } = new();

        // 儲存檔案標籤（檔名清單），用於生成結果時的條件判斷。
        public ObservableCollection<string> TagItems { get; } = new();


        // 允許上傳的副檔名清單（僅限 Word 與 PDF）
        private static readonly string[] AllowedExtensions = { ".doc", ".docx", ".pdf" };

        // MVVM Toolkit 自動產生屬性與通知
        [ObservableProperty] private bool isUploading; // 是否正在上傳
        [ObservableProperty] private string? currentUploadText; // 目前上傳中的檔名文字
        [ObservableProperty] private double currentUploadProgress; // 目前上傳進度（百分比）


        // 處理中狀態（生成結果時使用）
        private bool isProcessing;
        public bool IsProcessing
        {
            get => isProcessing;
            set => SetProperty(ref isProcessing, value);
        }

        // 遮蔽身份的設定
        private bool maskIdentity;
        public bool MaskIdentity
        {
            get => maskIdentity;
            set
            {
                SetProperty(ref maskIdentity, value);
                OnPropertyChanged(nameof(CanGenerate)); // 狀態改變時更新 CanGenerate
            }
        }

        // 遮蔽地址的設定
        private bool maskAddress;
        public bool MaskAddress
        {
            get => maskAddress;
            set
            {
                SetProperty(ref maskAddress, value);
                OnPropertyChanged(nameof(CanGenerate)); // 狀態改變時更新 CanGenerate
            }
        }


        /// <summary>
        /// 可以生成的條件：有檔案 + 勾任一遮蔽選項
        /// </summary>
        public bool CanGenerate => TagItems.Any() && (MaskIdentity || MaskAddress);


        // 上傳狀態對應的 UI 顯示控制。
        public Visibility UploadVisibility => IsUploading ? Visibility.Visible : Visibility.Collapsed;

        public UploadViewModel(PythonProcessor python)
        {
            _python = python;
            // 當標籤集合改變時，更新 CanGenerate 屬性
            TagItems.CollectionChanged += (_, __) => OnPropertyChanged(nameof(CanGenerate));
        }

        // 生成結果的指令。
        [RelayCommand]
        public async Task GenerateResultAsync()
        {
            // 如果條件不足，顯示警告對話框
            if (!CanGenerate)
            {
                ContentDialog warning = new()
                {
                    Title = "請檢查輸入條件",
                    Content = "請選擇至少一個檔案，並勾選一項遮蔽選項後才能生成結果。",
                    CloseButtonText = "我知道了",
                    XamlRoot = App.MainAppWindow.Content.XamlRoot
                };
                await warning.ShowAsync();
                return;
            }

            // 模擬處理中
            IsProcessing = true;
            await Task.Delay(2000);
            IsProcessing = false;


            // 導向到結果頁面
            if (App.MainAppWindow.Content is Frame frame)
            {
                frame.Navigate(typeof(ResultPage));
            }
        }


        // 選擇檔案的指令（使用 FileOpenPicker）。
        [RelayCommand]
        public async Task PickFilesAsync()
        {
            var picker = new FileOpenPicker();
            picker.FileTypeFilter.Add(".doc");
            picker.FileTypeFilter.Add(".docx");
            picker.FileTypeFilter.Add(".pdf");

            // 初始化檔案挑選器與視窗綁定
            var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(App.MainAppWindow);
            WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

            // 多檔選取
            var files = await picker.PickMultipleFilesAsync();
            if (files is null) return;

            foreach (var file in files)
                await HandleFileAsync(file);
        }


        // 拖曳檔案上傳的指令。
        [RelayCommand]
        public async Task DropFilesAsync(DragEventArgs e)
        {
            if (e.DataView.Contains(StandardDataFormats.StorageItems))
            {
                var items = await e.DataView.GetStorageItemsAsync();
                foreach (var item in items.OfType<StorageFile>())
                    await HandleFileAsync(item);
            }
        }

        // 處理檔案：檢查副檔名並加入上傳清單。
        public async Task HandleFileAsync(StorageFile file)
        {
            var ext = file.FileType.ToLower();
            if (!AllowedExtensions.Contains(ext))
            {
                await ShowUnsupportedFileDialogAsync(file.Name);
                return;
            }

            AddUploadItem(file);
        }


        // 顯示不支援檔案格式的提示。
        private async Task ShowUnsupportedFileDialogAsync(string filename)
        {
            var dialog = new ContentDialog
            {
                Title = "不支援的檔案格式",
                Content = $"檔案「{filename}」並非 Word 或 PDF 格式，請重新選擇。",
                CloseButtonText = "確定",
                XamlRoot = App.MainAppWindow.Content.XamlRoot
            };

            await dialog.ShowAsync();
        }

        // 新增檔案名稱到標籤清單。
        private void AddFileTag(string fileName)
        {
            if (!TagItems.Contains(fileName))
                TagItems.Add(fileName);
        }

        // 移除檔案與對應標籤。
        [RelayCommand]
        public void RemoveTag(string fileName)
        {
            var item = UploadItems.FirstOrDefault(f => f.FileName == fileName);
            if (item != null) UploadItems.Remove(item);
            TagItems.Remove(fileName);
        }


        // 新增檔案到上傳清單，並啟動模擬上傳流程。
        private void AddUploadItem(StorageFile file)
        {
            // 避免重複檔案
            if (UploadItems.Any(f => f.FileName == file.Name)) return;

            var item = new UploadFileInfo
            {
                FileName = file.Name,
                FullPath = file.Path,
                UploadProgress = 0,
                UploadStatus = "Waiting"
            };
            UploadItems.Add(item);
            AddFileTag(item.FileName);
            _ = SimulateUpload(item);
        }

        

        // 模擬檔案上傳的流程（假進度）。
        private async Task SimulateUpload(UploadFileInfo item)
        {
            IsUploading = true;
            CurrentUploadText = $"{item.FileName} uploading...";
            item.UploadStatus = "Uploading";

            for (int i = 0; i <= 100; i += 5)
            {
                await Task.Delay(50); // 模擬延遲
                item.UploadProgress = i;
                CurrentUploadProgress = i;
            }

            item.UploadStatus = "Upload complete";
            IsUploading = false;
        }

        // 丟給 Python 處理的指令。
        [RelayCommand]
        public async Task RunPythonAsync()
        {
            // 沒檔案會跳警告
            if (!UploadItems.Any())
            {
                await new ContentDialog
                {
                    Title = "沒有檔案",
                    Content = "請先選擇或拖曳檔案。",
                    CloseButtonText = "確定",
                    XamlRoot = App.MainAppWindow.Content.XamlRoot
                }.ShowAsync();
                return;
            }

            // 準備 Output 目錄
            var outDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                "AnoniMe", "outputs", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            Directory.CreateDirectory(outDir);

            // 組裝輸入檔與選項
            var inputs = UploadItems
                .Where(i => !string.IsNullOrEmpty(i.FullPath))
                .Select(i => new PythonFile(i.FileName, i.FullPath!))
                .ToList();

            var options = new Dictionary<string, object>
            {
                ["maskIdentity"] = MaskIdentity,
                ["maskAddress"] = MaskAddress
            };

            IsProcessing = true;
            CurrentUploadText = "Python 處理中...";
            CurrentUploadProgress = 0;

            try
            {
                var req = new PythonRequest(inputs, options, outDir);
                var result = await _python.ProcessAsync(
                    req,
                    progress: new Progress<int>(p => CurrentUploadProgress = p));

                // 回填各檔的 OutputPath / 狀態
                foreach (var r in result.Results)
                {
                    var item = UploadItems.FirstOrDefault(x =>
                        string.Equals(x.FullPath, r.Input, StringComparison.OrdinalIgnoreCase));
                    if (item is not null)
                    {
                        item.OutputPath = r.Output;
                        item.UploadStatus = r.Status; // e.g. "ok"
                    }
                }

                CurrentUploadText = "處理完成";
            }
            catch (Exception ex)
            {
                CurrentUploadText = $"Python 執行失敗：{ex.Message}";
            }
            finally
            {
                IsProcessing = false;
            }
        }

    }
}