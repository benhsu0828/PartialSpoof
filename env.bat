@echo off
REM === 設定根路徑為目前目錄 ===
set ROOT_PATH=%cd%
set XW_PATH=%ROOT_PATH%\project-NN-Pytorch-scripts.202102
set MODULES_PATH=%ROOT_PATH%\modules
set S3PRL_PATH=%MODULES_PATH%\s3prl

REM === 設定環境變數 ===
set PS_PATH=%ROOT_PATH%
set PYTHONPATH=%XW_PATH%;%ROOT_PATH%;%PYTHONPATH%;%MODULES_PATH%;%S3PRL_PATH%

echo 環境變數已設定完成
