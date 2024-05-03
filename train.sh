CUDA_VISIBLE_DEVICES=0 \
	accelerate launch \
	--num_processes 1 \
	--num_machines 1 \
	--mixed_precision fp16 \
	--main_process_ip 127.0.0.1 \
	--main_process_port 8868 \
	train_mash.py \
	model=zigzag8_b1_pe2 \
	use_latent=1 \
	data=celebamm256_uncond \
	ckpt_every=100 \
	data.sample_fid_n=5_000 \
	data.sample_fid_bs=4 \
	data.sample_fid_every=10_000 \
	data.batch_size=4 \
	note=_
