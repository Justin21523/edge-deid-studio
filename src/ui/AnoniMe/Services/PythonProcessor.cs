using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace AnoniMe.Services
{

    public record PythonFile(string Name, string Path);
    public record PythonRequest(List<PythonFile> InputFiles, Dictionary<string, object> Options, string OutputDir);
    public record PythonFileResult(string Input, string Output, string Status);
    public record PythonResult(List<PythonFileResult> Results);


    /// <summary>
    /// C# ↔ Python 橋接器：以 stdin 傳入 JSON、以 stdout 取回 JSON，stderr 讀取進度。
    /// </summary>
    public sealed class PythonProcessor
    {
        private string _pythonExe;
        private string _scriptPath;

        public PythonProcessor(string pythonExe, string scriptPath)
        {
            _pythonExe = pythonExe;
            _scriptPath = scriptPath;
        }

        public async Task<PythonResult> ProcessAsync(
            PythonRequest request,
            IProgress<int>? progress = null,
            CancellationToken ct = default)
        {
            // 這裡實作呼叫 Python 腳本的邏輯
            // 例如使用 Process 啟動 Python 執行檔並傳入參數
            // 然後解析輸出結果，轉換為 PythonResult

            var psi = new ProcessStartInfo
            {
                FileName = _pythonExe,
                ArgumentList = { _scriptPath },
                RedirectStandardInput = true, // 可以丟資料
                RedirectStandardOutput = true, //可以接收結果
                RedirectStandardError = true, //可接收錯誤訊離
                UseShellExecute = false, //不要透過 Windows Shell 來啟動程式，而是直接用程式路徑啟動
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };


            // 進度條的部分
            using var p = new Process { StartInfo = psi, EnableRaisingEvents = true };

            // 解析 Python 回報的進度：stderr 的 "PROGRESS:NN"
            p.ErrorDataReceived += (s, e) =>
            {
                if (e.Data is null) return;
                if (e.Data.StartsWith("PROGRESS:") &&
                    int.TryParse(e.Data.AsSpan("PROGRESS:".Length), out var val))
                {
                    progress?.Report(val);
                }
            };

            try
            {
                if (!p.Start())
                    throw new InvalidOperationException("無法啟動 Python 處理程序。");

                p.BeginErrorReadLine();

                // 傳入 JSON 請求
                await p.StandardInput.WriteAsync(JsonSerializer.Serialize(request));
                await p.StandardInput.FlushAsync();
                p.StandardInput.Close();

                // 讀回 JSON 響應
                string json = await p.StandardOutput.ReadToEndAsync();

                // 等待結束（支援取消）
                await Task.Run(() => p.WaitForExit(), ct);

                if (p.ExitCode != 0)
                {
                    // 嘗試把 stderr 全收（EndErrorReadLine 會在進程結束後自動結束）
                    throw new InvalidOperationException($"Python 處理失敗，ExitCode={p.ExitCode}。");
                }

                if (string.IsNullOrWhiteSpace(json))
                    throw new InvalidOperationException("Python 未回傳任何資料。");

                var result = JsonSerializer.Deserialize<PythonResult>(
                    json,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                if (result is null)
                    throw new InvalidOperationException("回傳資料不是有效的 JSON 結構（PythonResult）。");

                return result;
            }
            finally
            {
                try { if (!p.HasExited) p.Kill(true); } catch { /* ignore */ }
            }
        }
    
    }
}
