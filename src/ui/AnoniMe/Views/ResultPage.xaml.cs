using AnoniMe.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Navigation;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;

namespace AnoniMe.Views;

public sealed partial class ResultPage : Page
{
    public ResultViewModel ViewModel { get; } = App.HostContainer.Services.GetService<ResultViewModel>();

    public ResultPage()
    {
        this.InitializeComponent();
        this.DataContext = ViewModel;
    }
}
