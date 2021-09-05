CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --synthetic_train_data_dir /data1/zem/Resnet.CRNN/data/lmdb/recognition/CVPR2016/ /data1/zem/Resnet.CRNN/data/lmdb/recognition/NIPS2014/ \
  --test_data_dir /data1/zem/Resnet.CRNN/data/lmdb/recognition/benchmark_lmdbs_new/IIIT5K_3000/ \
  --batch_size 100 \
  --workers 4 \
  --height 64 \
  --width 256 \
  --voc_type ALLCASES_SYMBOLS \
  --arch ResNet_ASTER \
  --with_lstm \
  --logs_dir /data1/zem/aster.pytorch/logs/baseline_aster \
  --real_logs_dir /data1/zem/aster.pytorch/logs \
  --max_len 100 \
  --STN_ON \
  --tps_inputsize 32 64 \
  --tps_outputsize 32 100 \
  --tps_margins 0.05 0.05 \
  --stn_activation none \
  --num_control_points 20 \