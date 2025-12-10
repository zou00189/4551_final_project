#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from data_preprocess import VehicleStateDataset
from model.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

dataset_dir = "episodes"
checkpoint_dir = "checkpoints" 

obs_horizon = 2
pred_horizon = 16
action_horizon = 8

batch_size = 64
num_epochs = 801
learning_rate = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create checkpoint directory if it doesn't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# create Dataset
dataset = VehicleStateDataset(
    dataset_dir=dataset_dir,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

obs_dim = len(dataset.obs_keys)          # 6 dims: obstacle xy, box xy, vehicle xy
action_dim = len(dataset.action_keys)    # 2 dims: vx, vy

print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")


noise_pred_net = ConditionalUnet1D(
    input_dim = action_dim,
    global_cond_dim = obs_dim * obs_horizon,
).to(device)

# Diffusion noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2",
    clip_sample=False,
    prediction_type='epsilon'
)

# EMA model
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75
)

optimizer = torch.optim.AdamW(
    noise_pred_net.parameters(),
    lr=learning_rate,
    weight_decay=1e-6
)

lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

# training
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    epoch_losses = []

    for batch in tqdm(dataloader, desc="Batch", leave=False):
        nobs = batch["obs"].to(device)        # (B, obs_horizon, obs_dim)
        naction = batch["action"].to(device)  # (B, pred_horizon, action_dim)
        B = nobs.shape[0]

        # FiLM conditioning: flatten obs horizon
        obs_cond = nobs[:, :obs_horizon, :].reshape(B, -1) # (B, obs_horizon * obs_dim)

        # sample noise
        noise = torch.randn_like(naction, device=device)

        # random timestep per sample
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # forward diffusion: add noise
        noisy_action = noise_scheduler.add_noise(
            naction, noise, timesteps
        )

        # predict noise
        noise_pred = noise_pred_net(
            noisy_action, timesteps, global_cond=obs_cond
        )

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        ema.step(noise_pred_net.parameters())

        epoch_losses.append(loss.item())

    avg_loss = np.mean(epoch_losses)
    print(f"Epoch {epoch+1}/{num_epochs} Loss = {avg_loss:.6f}")

    if epoch % 50 == 0 and epoch >= 100:

        ema_net = noise_pred_net
        ema.copy_to(ema_net.parameters())
        save_path = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch+1}.pth")

        torch.save({
            "model_state_dict": ema_net.state_dict(),
            "stats": dataset.stats,
            "config": {
                "obs_horizon": obs_horizon,
                "pred_horizon": pred_horizon,
                "action_horizon": action_horizon,
                "obs_dim": obs_dim,
                "action_dim": action_dim
            }
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

print("Training complete")


