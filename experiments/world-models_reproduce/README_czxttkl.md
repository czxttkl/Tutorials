My reproduce step:
1. run data/carracing.py
I set some arguments' default values: 
dir=
rollouts=126, meaning that I'll collect 126 .npz files in dir, each file containing
one rollout, one rollout containing 1000 steps (as specified by seq_len in carracing.py)

2. run trainvae.py
I set vaelogdir argument, which is the directory VAE model will save to
I also set datadir, which is the directory of carracing transition data
you also need to pay attention to train_size when initializing RolloutObservationDataset,
this is the number of files you want to use for training (the rest files will be used for testing).

3. run trainmdrnn.py
I set train_size in RolloutSequenceDataset, logdir is the directory where mdrnn is output 
(I set the same as vae's output directory (vaelogdir in step 2))