export CUDA_VISIBLE_DEVICES=7
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
SAMPLE_FLAGS="--batch_size 8 --num_samples 96 --timestep_respacing 250"
python scripts/image_sample.py $MODEL_FLAGS $SAMPLE_FLAGS $@
