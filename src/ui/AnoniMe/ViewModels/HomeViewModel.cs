using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using System;
using System.Threading.Tasks;

namespace AnoniMe.ViewModels
{  
    /// <summary>
    /// 首頁的 ViewModel，負責處理首頁上的動畫效果與頁面跳轉。
    /// </summary>
    public partial class HomeViewModel : ObservableObject
    {

        // 文字動畫
        [ObservableProperty]
        private string? typingText;

        // 點點動畫
        [ObservableProperty]
        private string? dotsText;        

        // 計時器
        private DispatcherTimer? _dotsTimer;

        // 控制 dot 數量
        private int _dotIndex = 0;

        // 預先定義 dot 變化
        private readonly string[] _dotCycle = new[] { "", ".", "..", "..." };

        // 建構函式
        public HomeViewModel()
        {
            // 避免一開始就跑完動畫，不放在這裡
        }

        // 啟動首頁動畫（打字機效果 + 點點動畫）。
        public void StartAnimations(int typingDelay = 200)
        {
            _ = ShowTypingEffectAsync("EASILY LIVE WITH SAFETY", typingDelay);
            StartDotsAnimation();
        }


        // 模擬打字機效果，逐字顯示文字。
        private async Task ShowTypingEffectAsync(string text, int delayMs = 100)
        {
            typingText = "";
            foreach (char c in text)
            {
                typingText += c;
                await Task.Delay(delayMs);
            }
        }

        // 啟動點點動畫（每 400ms 切換一次內容）。
        private void StartDotsAnimation()
        {
            _dotsTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromMilliseconds(400)
            };

            _dotsTimer.Tick += (s, e) =>
            {
                dotsText = _dotCycle[_dotIndex];
                _dotIndex = (_dotIndex + 1) % _dotCycle.Length;
            };

            _dotsTimer.Start();
        }

        // 點擊「上傳」按鈕時執行的命令（會綁定在 Button 上）
        [RelayCommand]
        private void Upload()
        {
            // Use App.MainAppWindow instead of App.Current.Windows[0]
            if (App.MainAppWindow?.Content is Frame rootFrame)
            {
                rootFrame.Navigate(typeof(UploadPage)); // Navigate to UploadPage
            }
        }
    }
}
