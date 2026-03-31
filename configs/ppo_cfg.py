from skrl.resources.preprocessors.torch import RunningStandardScaler

ppo_cfg = {
    "rollouts": 2048,
    "learning_epochs": 8,
    "mini_batches": 4,
    "discount_factor": 0.99,
    "lambda": 0.95,
    "learning_rate": 3e-4,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "grad_norm_clip": 0.5,
    "entropy_loss_scale": 0.01,
    "value_loss_scale": 0.5,
    "kl_threshold": 0,
    "state_preprocessor": RunningStandardScaler,
    "state_preprocessor_kwargs": {},
    "value_preprocessor": RunningStandardScaler,
    "value_preprocessor_kwargs": {},
    "random_timesteps": 0,
    "learning_starts": 0,
    "experiment": {
        "directory": "runs/ppo_recon",
        "experiment_name": "",
        "write_interval": 1000,
        "checkpoint_interval": 50000,
        "wandb": True,
        "wandb_kwargs": {
            "project": "OGE-Recon",
            "name": "ppo-recon-task",
            "tags": ["ppo", "reconnaissance"],
        },
    },
}

trainer_cfg = {
    "timesteps": 8_000_000,
    "headless": True,
}