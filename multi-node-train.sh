MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 512 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--batch_size 6 --lr 2e-5 --save_interval 1000 --weight_decay 0.00"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
deepspeed --hostfile=hostfile.txt /workspace/guided_diffusion_deepspeeed/scripts/image_train.py --deepspeed_config ds_config.json --data_dir /data1/songtao.liu/datasets/feitian/ --deepspeed $MODEL_FLAGS $TRAIN_FLAGS $DIFFUSION_FLAGS $@
