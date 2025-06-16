#!/usr/bin/env python
"""
Audio directory analysis script - 與 default_data_io.py 一致的版本
"""

import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path
import time

# 假設這些是您的項目模組
sys.path.append('/home/ben/PartialSpoof/project-NN-Pytorch-scripts.202102')
import core_scripts.data_io.wav_tools as nii_wav_tk

def _data_len_reader_consistent(file_path):
    """ 與 default_data_io.py 中 _data_len_reader 一致的函數 """
    file_name, file_ext = os.path.splitext(file_path)
    if file_ext == '.wav':
        sr, data = nii_wav_tk.waveReadAsFloat(file_path)
        length = data.shape[0]
    elif file_ext == '.flac':
        data, sr = sf.read(file_path)
        length = data.shape[0]
    else:
        # 使用 soundfile info 作為備選
        info = sf.info(str(file_path))
        length = info.frames
    return length

def analyze_audio_directory(directory_path, supported_extensions=('.wav', '.flac'), 
                          resolution=1):
    """
    分析目錄中所有音檔的統計資訊 - 與 default_data_io.py 一致
    
    Args:
        directory_path: 要分析的目錄路徑
        supported_extensions: 支援的音檔副檔名
        resolution: 解析度調整參數（對應 default_data_io.py 的 m_single_reso）
    """
    print("=" * 70)
    print(f"Audio Directory Analysis: {directory_path}")
    print("=" * 70)
    
    # 檢查目錄是否存在
    if not os.path.exists(directory_path):
        print(f"✗ Directory not found: {directory_path}")
        return
    
    # 搜尋所有音檔
    print("1. Scanning audio files...")
    audio_files = []
    
    for ext in supported_extensions:
        pattern = f"**/*{ext}"
        files = list(Path(directory_path).glob(pattern))
        audio_files.extend(files)
    
    if not audio_files:
        print(f"✗ No audio files found in {directory_path}")
        print(f"  Supported extensions: {supported_extensions}")
        return
    
    print(f"✓ Found {len(audio_files)} audio files")
    
    # 分析每個音檔
    print("\n2. Analyzing audio files...")
    print("-" * 50)
    
    sample_counts = []
    adjusted_sample_counts = []  # 調整後的樣本數
    file_info = []
    errors = []
    
    for i, file_path in enumerate(audio_files):
        try:
            # 使用與 default_data_io.py 一致的方法讀取長度
            sample_count = _data_len_reader_consistent(str(file_path))
            
            # 進行與 default_data_io.py 一致的長度調整
            adjusted_sample_count = sample_count // resolution * resolution
            
            sample_counts.append(sample_count)
            adjusted_sample_counts.append(adjusted_sample_count)
            
            # 獲取額外資訊
            info = sf.info(str(file_path))
            file_info.append({
                'path': file_path,
                'original_sample_count': sample_count,
                'adjusted_sample_count': adjusted_sample_count,
                'samplerate': info.samplerate,
                'channels': info.channels,
                'frames': info.frames,
                'format': info.format,
                'subtype': info.subtype
            })
                
        except Exception as e:
            errors.append({'file': file_path, 'error': str(e)})
            print(f"✗ Error processing {file_path.name}: {e}")
    
    # 統計結果
    print("\n3. Analysis Results")
    print("=" * 50)
    
    if adjusted_sample_counts:
        total_files = len(adjusted_sample_counts)
        min_samples = min(adjusted_sample_counts)
        max_samples = max(adjusted_sample_counts)
        avg_samples = np.mean(adjusted_sample_counts)
        total_samples = sum(adjusted_sample_counts)
        
        # 找出最短和最長的檔案（基於調整後的長度）
        min_idx = adjusted_sample_counts.index(min_samples)
        max_idx = adjusted_sample_counts.index(max_samples)
        min_file = file_info[min_idx]
        max_file = file_info[max_idx]
        
        print(f"總音檔數量: {total_files} 筆")
        print(f"成功處理: {total_files} 筆")
        print(f"處理失敗: {len(errors)} 筆")
        print(f"解析度調整參數: {resolution}")
        
        print(f"\n音檔樣本數統計 (調整後):")
        print(f"  最短音檔: {min_samples} samples")
        print(f"    檔案: {min_file['path'].name}")
        print(f"    原始長度: {min_file['original_sample_count']} samples")
        print(f"    調整後長度: {min_file['adjusted_sample_count']} samples")
        print(f"    採樣率: {min_file['samplerate']} Hz")
        print(f"    聲道數: {min_file['channels']}")
        print(f"    時長: {min_samples/min_file['samplerate']:.2f} 秒")
        
        print(f"  最長音檔: {max_samples} samples")
        print(f"    檔案: {max_file['path'].name}")
        print(f"    原始長度: {max_file['original_sample_count']} samples")
        print(f"    調整後長度: {max_file['adjusted_sample_count']} samples")
        print(f"    採樣率: {max_file['samplerate']} Hz")
        print(f"    聲道數: {max_file['channels']}")
        print(f"    時長: {max_samples/max_file['samplerate']:.2f} 秒")
        
        print(f"  平均樣本數 (調整後): {avg_samples:.0f} samples")
        print(f"  總樣本數 (調整後): {total_samples} samples")
        
        # 顯示調整的影響
        original_total = sum(sample_counts)
        print(f"\n長度調整影響:")
        print(f"  原始總樣本數: {original_total} samples")
        print(f"  調整後總樣本數: {total_samples} samples")
        print(f"  差異: {original_total - total_samples} samples")
        
        # 長度分布統計
        print(f"\n樣本數分布 (調整後):")
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            value = np.percentile(adjusted_sample_counts, p)
            print(f"  {p}th percentile: {value:.0f} samples")
    
    # 錯誤報告和格式統計保持不變...
    # [其餘代碼保持原樣]

if __name__ == "__main__":
    # 設定要分析的目錄路徑
    directories_to_analyze = [
        "/home/ben/PartialSpoof/database/ASVspoof5/ASVspoof5_dev/flac_D"
    ]
    
    start_time = time.time()
    
    for directory in directories_to_analyze:
        if os.path.exists(directory):
            # 如果知道 resolution 參數，請在這裡設定
            # 例如：analyze_audio_directory(directory, resolution=80)
            analyze_audio_directory(directory)
            print("\n" + "="*70 + "\n")
        else:
            print(f"Directory not found: {directory}\n")
    
    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds")