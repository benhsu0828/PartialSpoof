#!/bin/bash

#Example to measure different types of EER

PS_PATH=/home/ben/PartialSpoof/
pred_DIR=/home/ben/PartialSpoof/03multireso/single-reso/utt/01 	# predicted dir.
dset=dev 	# dev eval

#############Utterance-level EER
ASV_SCORES_FILE=${PS_PATH}"/database/protocols/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv."$dset".gi.trl.scores.txt"
python ${PS_PATH}/metric/UtteranceEER.py \
    --pred_file ${pred_DIR}/to_dir/log_${dset} \
    --asv_score_file ${ASV_SCORES_FILE} \
    --utt2label_file ${PS_PATH}/database/protocols/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm."$dset".trl.txt