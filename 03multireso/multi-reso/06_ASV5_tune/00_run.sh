#!/bin/bash

stage=$1
CON_PATH=../../../database
OUTPUT_DIR=output
PRETRAINED_MODEL=trained_network.pt

if [ ! -d ${OUTPUT_DIR}  ]; then
    mkdir ${OUTPUT_DIR}
fi

#stage 0:
if [ $stage -le 0 ]; then
    ssl_link="https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv.pt"
    if [ ! -f ../../modules/ssl_pretrain/w2v_large_lv_fsh_swbd_cv.pt  ]; then
        wget -q --show-progress -c ${ssl_link} -O ../../modules/ssl_pretrain
    fi
fi

#stage 1:
if [ $stage -le 1 ]; then
    python main.py --module-model model --model-forward-with-file-name --seed 87 \
            --ssl-finetune \
            --multi-scale-active utt \
            --num-workers 12 --epochs 50 --no-best-epochs 5 --batch-size 16 \
            --sampler block_shuffle_by_length --lr-decay-factor 0.8 --lr-scheduler-type 1 \
            --lr 0.000001 \
            --module-config config_ps.config_train_on_5dev \
            --data-type asvspoof \
            --save-model-dir ./checkpoints/ \
            --save-trained-name best_model \
            --trained-model ${PRETRAINED_MODEL} \
            > ${OUTPUT_DIR}/log_finetune 2> ${OUTPUT_DIR}/log_err
fi

#stage 2
if [ $stage -le 2 ]; then
    python main.py --inference --module-model model --model-forward-with-file-name --module-config config_ps.config_test_on_5dev  \
       --output-dir ${OUTPUT_DIR}/dev > ${OUTPUT_DIR}/log_output_dev 2>&1 \
        --trained-model /home/ben/PartialSpoof/03multireso/multi-reso/06_ASV5_tune/checkpoints/best_model.pt \
       --num-workers 4 \
       --batch-size 4 \
       --data-type asvspoof &

    python main.py --inference --module-model model --model-forward-with-file-name  --module-config config_ps.config_test_on_5eval\
       --output-dir ${OUTPUT_DIR}/eval > ${OUTPUT_DIR}/log_output_eval 2>&1 \
        --trained-model /home/ben/PartialSpoof/03multireso/multi-reso/06_ASV5_tune/checkpoints/best_model.pt \
         --num-workers 4 \
         --batch-size 4 \
         --data-type asvspoof 
fi