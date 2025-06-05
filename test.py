#!/usr/bin/env python
"""
Audio file comparison test script
"""

import os
import sys
import numpy as np
import soundfile as sf

# 假設這些是您的項目模組
sys.path.append('/home/ben/PartialSpoof/project-NN-Pytorch-scripts.202102')

import core_scripts.data_io.wav_tools as nii_wav_tk

def compare_audio_files(wav_file_path, flac_file_path):
    """
    比較 WAV 和 FLAC 檔案的差異
    """
    print("=" * 60)
    print("Audio File Comparison")
    print("=" * 60)
    
    # 讀取 WAV 檔案 (使用原始方法)
    print("\n1. Reading WAV file with nii_wav_tk...")
    try:
        sr_wav, data_wav = nii_wav_tk.waveReadAsFloat(wav_file_path)
        print(f"✓ WAV file loaded successfully")
        print(f"  File: {wav_file_path}")
        print(f"  Sample rate: {sr_wav} Hz")
        print(f"  Data shape: {data_wav.shape}")
        print(f"  Data type: {data_wav.dtype}")
        print(f"  Data range: [{data_wav.min():.6f}, {data_wav.max():.6f}]")
        print(f"  Duration: {len(data_wav) / sr_wav:.2f} seconds")
    except Exception as e:
        print(f"✗ Error reading WAV: {e}")
        return
    
    # 讀取 FLAC 檔案 (使用 soundfile)
    print("\n2. Reading FLAC file with soundfile...")
    try:
        data_flac, sr_flac = sf.read(flac_file_path)
        print(f"✓ FLAC file loaded successfully")
        print(f"  File: {flac_file_path}")
        print(f"  Sample rate: {sr_flac} Hz")
        print(f"  Data shape: {data_flac.shape}")
        print(f"  Data type: {data_flac.dtype}")
        print(f"  Data range: [{data_flac.min():.6f}, {data_flac.max():.6f}]")
        print(f"  Duration: {len(data_flac) / sr_flac:.2f} seconds")
        
        # 處理多聲道
        if data_flac.ndim > 1:
            print(f"  Multi-channel detected, converting to mono...")
            data_flac = data_flac.mean(axis=1)
            print(f"  New shape after mono conversion: {data_flac.shape}")
            
    except Exception as e:
        print(f"✗ Error reading FLAC: {e}")
        return
    
    # 比較差異
    print("\n3. Comparing differences...")
    print("-" * 40)
    
    # 採樣率比較
    print(f"Sample rate difference: {sr_wav - sr_flac} Hz")
    if sr_wav != sr_flac:
        print("⚠ WARNING: Sample rates are different!")
    else:
        print("✓ Sample rates match")
    
    # 長度比較
    len_diff = len(data_wav) - len(data_flac)
    print(f"Length difference: {len_diff} samples")
    if abs(len_diff) > 10:  # 允許小幅差異
        print("⚠ WARNING: Lengths are significantly different!")
    else:
        print("✓ Lengths are similar")
    
    # 數據範圍比較
    wav_range = data_wav.max() - data_wav.min()
    flac_range = data_flac.max() - data_flac.min()
    range_diff = abs(wav_range - flac_range)
    print(f"Data range difference: {range_diff:.6f}")
    
    # 如果長度相同，計算數值差異
    if len(data_wav) == len(data_flac):
        print("\n4. Detailed numerical comparison...")
        print("-" * 40)
        
        # 計算差異統計
        diff = np.abs(data_wav - data_flac)
        print(f"Mean absolute difference: {diff.mean():.8f}")
        print(f"Max absolute difference: {diff.max():.8f}")
        print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.8f}")
        
        # 相關性
        correlation = np.corrcoef(data_wav, data_flac)[0, 1]
        print(f"Correlation coefficient: {correlation:.8f}")
        
        if diff.max() < 1e-6:
            print("✓ Files are essentially identical")
        elif diff.max() < 1e-3:
            print("✓ Files are very similar (small differences)")
        else:
            print("⚠ Files have noticeable differences")
    else:
        print("\n4. Cannot compare values (different lengths)")
        
        # 比較前 N 個樣本
        min_len = min(len(data_wav), len(data_flac))
        if min_len > 1000:
            print(f"Comparing first {min_len} samples...")
            diff = np.abs(data_wav[:min_len] - data_flac[:min_len])
            print(f"Mean absolute difference (first {min_len}): {diff.mean():.8f}")
            print(f"Max absolute difference (first {min_len}): {diff.max():.8f}")

def test_specific_files():
    """
    測試特定的音檔
    """
    # 設定檔案路徑 - 請修改為您的實際路徑
    wav_file = "/home/ben/PartialSpoof/database/ASVspoof2019_LA_dev/wav/LA_D_1000265.wav"
    flac_file = "/home/ben/PartialSpoof/database/ASVspoof2019_LA_dev/flac/LA_D_1000265.flac"
    
    # 檢查檔案是否存在
    if not os.path.exists(wav_file):
        print(f"WAV file not found: {wav_file}")
        return
    
    if not os.path.exists(flac_file):
        print(f"FLAC file not found: {flac_file}")
        return
    
    # 執行比較
    compare_audio_files(wav_file, flac_file)

def test_soundfile_only():
    """
    測試純 soundfile 讀取兩種格式
    """
    print("\n" + "=" * 60)
    print("Testing soundfile with both formats")
    print("=" * 60)
    
    wav_file = "/home/ben/PartialSpoof/database/ASVspoof2019_LA_dev/wav/LA_D_1000265.wav"
    flac_file = "/home/ben/PartialSpoof/database/ASVspoof2019_LA_dev/flac/LA_D_1000265.flac"
    
    try:
        # 用 soundfile 讀取 WAV
        data_wav_sf, sr_wav_sf = sf.read(wav_file)
        print(f"WAV (soundfile): shape={data_wav_sf.shape}, sr={sr_wav_sf}, dtype={data_wav_sf.dtype}")
        
        # 用 soundfile 讀取 FLAC
        data_flac_sf, sr_flac_sf = sf.read(flac_file)
        print(f"FLAC (soundfile): shape={data_flac_sf.shape}, sr={sr_flac_sf}, dtype={data_flac_sf.dtype}")
        
        # 比較
        if data_wav_sf.shape == data_flac_sf.shape:
            diff = np.abs(data_wav_sf - data_flac_sf)
            print(f"Soundfile comparison - Max diff: {diff.max():.8f}")
        
    except Exception as e:
        print(f"Error in soundfile test: {e}")

if __name__ == "__main__":
    # 執行測試
    test_specific_files()
    test_soundfile_only()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)