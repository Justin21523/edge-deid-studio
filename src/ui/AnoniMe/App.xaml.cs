using AnoniMe.Services;
using AnoniMe.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.UI.Xaml;
using System;
using System.IO;

namespace AnoniMe
{
  
    public partial class App : Application
    {
        // 註冊 DI 容器
        public static IHost? HostContainer { get; private set; }

        public static Window? MainAppWindow { get; private set; }


        public App()
        {
            InitializeComponent();

        }

        // 啟動時呼叫
        protected override void OnLaunched(Microsoft.UI.Xaml.LaunchActivatedEventArgs args)
        {
            RegisterComponents();
            MainAppWindow = new MainWindow();
            MainAppWindow.Activate();

        }

        // 設定容器、註冊服務
        private void RegisterComponents()
        {
            HostContainer = Host.CreateDefaultBuilder()
                .ConfigureServices(services =>
                {

                    // 路徑依你的專案調整
                    var pythonExe = @".venv\Scripts\python.exe";
                    //var scriptPath = Path.Combine(AppContext.BaseDirectory, "Scripts", "processor.py");
                    //services.AddSingleton(new PythonProcessor(pythonExe, scriptPath));

                    // 註冊 MainViewModel，每次呼叫都會建立新的實例 (Transient)
                    services.AddTransient<HomeViewModel>();
                    services.AddTransient<UploadViewModel>();
                    services.AddTransient<ResultViewModel>();
                    // 其他服務註冊...
                }). Build();
        }
    }
}
