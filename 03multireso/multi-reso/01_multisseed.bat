@echo off
REM 啟動錯誤中斷（模擬 set -e）
setlocal enabledelayedexpansion

REM 取得參數
set seed_pow=%1
if "%seed_pow%"=="" (
    echo 請輸入 seed_pow，例如：2
    exit /b 1
)

REM 計算 seed = 10^(seed_pow - 1)
set /a pow=%seed_pow%-1

REM 使用 powershell 計算指數 (因為 Windows 批次檔無法直接做 10 的乘冪)
for /f %%i in ('powershell -command "Write-Output ([math]::Pow(10, %pow%))"') do set seed=%%i

REM 設定資料夾名稱
set base_name=exp-
set base_dir=base
set name=%base_name%%seed_pow%

REM 刪除目標資料夾，重新複製
rmdir /s /q %name%
xcopy /e /i /y %base_dir% %name%

REM 進入新資料夾
cd %name%

REM 修改 00_run.sh 中的 seed 值
powershell -Command "(Get-Content 00_run.sh) -replace '--seed 1 ', '--seed %seed% ' | Set-Content 00_run.sh"

REM 執行腳本
bash 00_run.sh 0

REM 返回上一層
cd ..

endlocal
