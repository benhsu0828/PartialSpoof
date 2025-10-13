#!/usr/bin/python

"""
Used to evaluate segment model in utterance-level detection
input: output/log_output_dev
       uttscore are in seg1.min, seg2.min, seg3.min, seg4.min,...,utt
usage: 
print: Utterance-level eer for each scales.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sandbox import eval_asvspoof
np.set_printoptions(linewidth=100000)



parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--pred_file',type=str, default='')
parser.add_argument('--asv_score_file',type=str, default='To calculate t-DCF.')
parser.add_argument('--utt2label_file',type=str, default='../../../database/ASVspoof2019.LA.cm.dev.trl.txt')
        #ASV_SCORES_FILE="{}/database/protocols/PartialSpoof_LA_asv_scores/PartialSpoof.LA.asv.{}.gi.trl.scores.txt".format(PS_PATH, dset)
        # ZhangLin: But I personally do not recommend using t-DCF, because Partial Spoof is designed 
        # not only for ASVspoof, which aims to deceive machines, 
        # but also for DeepFake, which is intended to fool humans.
parser.add_argument('--output_csv',type=str, default='', help='Output CSV file path (optional)') 
args = parser.parse_args()


#######Configuration.
PS_PATH = "/home/smg/zhanglin/workspace/PROJ/Public/CODE/PartialSpoof"
print_mean = True #whetehr print mean of EERs for all random seeds 
print_each = True #whetehr print each EER for all random seeds 
SCORE_TYPE='min'
utt2label=dict([ [line.split()[1].strip(), (line.split()[4]).strip()] for line in open(args.utt2label_file) ])
print(f"DEBUG: Loaded {len(utt2label)} utterances from {args.utt2label_file}")

# 輸出 utt2label 到 CSV
def save_utt2label_to_csv():
    """將 utt2label 字典保存為 CSV 檔案"""
    
    # 準備資料
    data = []
    for filename, label in utt2label.items():
        data.append({
            'filename': filename,
            'label': label
        })
    
    # 建立 DataFrame
    df = pd.DataFrame(data)
    
    # 決定輸出檔案路徑
    if args.output_csv:
        csv_file = args.output_csv
    else:
        # 根據 pred_file 自動生成檔名
        base_name = os.path.splitext(os.path.basename("./check"))[0]
        csv_file = f"{base_name}_utt2label.csv"
    
    # 確保輸出目錄存在
    output_dir = os.path.dirname(csv_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存 CSV
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved utt2label to: {csv_file}")
    print(f"📊 Total entries: {len(df)}")
    print(f"📈 Label distribution:")
    print(df['label'].value_counts().to_string())
    
    return csv_file

# 保存 utt2label
# utt2label_csv = save_utt2label_to_csv()

scale_num = 7
MAX_col = scale_num + 3  #9 3+6
def parse_txt(file_path, col):
    """
    create score lists for bona fide and spoof.
    """
    bonafide = []
    spoofed = []
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            if line.startswith('Output,'):
                temp = line.rstrip('\n').split(',')
                flag = utt2label[temp[1].strip()]
                if (flag == 'bonafide'):
                    bonafide.append(float(temp[col]))
                else:
                    #spoofed.append(pow(float(temp[col]),2))
                    spoofed.append(float(temp[col]))
    bonafide = np.array(bonafide)
    spoofed = np.array(spoofed)

    #####
    # 加入除錯資訊
    # print(f"\nDEBUG Col {col}:")
    print(f"  Bonafide count: {len(bonafide)}, Spoof count: {len(spoofed)}")
    # print(f"  Bonafide: min={bonafide.min():.4f}, max={bonafide.max():.4f}, mean={bonafide.mean():.4f}")
    # print(f"  Spoof: min={spoofed.min():.4f}, max={spoofed.max():.4f}, mean={spoofed.mean():.4f}")
    # print(f"  Expected: bonafide > spoof for good performance")
    if bonafide.mean() < spoofed.mean():
        print(f"  ⚠️  WARNING: Bonafide scores < Spoof scores (may need to flip)")
    #####

    return bonafide, spoofed



if __name__ == "__main__":

    # print(f"DEBUG: Reading ASV file: {args.asv_score_file}")

    print(args.pred_file)
    mintDCF_oneseed_cols = []
    eer_oneseed_cols = []
    threshold_oneseed_cols =[]

    for col in np.arange(3, MAX_col):
        bonafide, spoofed = parse_txt(args.pred_file, col)

        # 只計算 EER
        eer, threshold = eval_asvspoof.compute_eer(bonafide, spoofed)
        mintDCF = 0.0  # 設定為預設值

        mintDCF_oneseed_cols.append(mintDCF)
        eer_oneseed_cols.append(eer)
        threshold_oneseed_cols.append(threshold)

    if (print_each):
        print('===' + str(args.pred_file) + '===')
        print(np.array(eer_oneseed_cols) * 100)
        print('---threshold---')
        print(np.array(threshold_oneseed_cols))
                
