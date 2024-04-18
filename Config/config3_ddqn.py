

config = {
    "env_name": "ALE/Assault-v5",
    "gamma": 0.99,
    "seed": 2157,
    "buffer_size": 20_000,
    "timesteps_per_epoch": 4,
    "batch_size": 32,
    "total_steps": 3_000_000,
    "decay_steps": 750_000, #steps before epsilon stops decreasing
    "decay_steps_beta": 500_000, #steps before epsilon stops decreasing
    "learning_rate": 1e-5 * 25,
    "init_epsilon": 1,
    "final_epsilon": 0.1,
    "loss_freq": 50, # steps to show loss  
    "refresh_target_network_freq": 5000,
    "eval_freq": 5000, # to follow the procces
    'W':64,
    'H':64,
    'alpha': 0.6,
    "init_beta": 0.4,
    "final_beta": 1,
    "loading": True,
    "steps_per_save": 5000,
    "file_name": "dqn_with_exp_replay_skip_truly_1.pth",
    "skip": 1
}