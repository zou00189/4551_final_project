import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os

class VehicleStateDataset(Dataset):
    def __init__(self, dataset_dir, pred_horizon, obs_horizon, action_horizon, stats=None):
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.stats = stats
        
        self.column_names = [
            "timestamp", 
            "vehicle_x", "vehicle_y", "vehicle_z", "vehicle_qx", "vehicle_qy", "vehicle_qz", "vehicle_qw", 
            "vehicle_vx", "vehicle_vy", "vehicle_wz", 
            "obstacle_x", "obstacle_y", "obstacle_z", "obstacle_qx", "obstacle_qy", "obstacle_qz", "obstacle_qw", 
            "obstacle_vx", "obstacle_vy", "obstacle_wz", 
            "box_x", "box_y", "box_z", "box_qx", "box_qy", "box_qz", "box_qw"
        ]

        # Obs: obstacle x,y | object(box) x,y | vehicle x,y
        self.obs_keys = ['obstacle_x', 'obstacle_y', 'box_x', 'box_y', 'vehicle_x', 'vehicle_y']
        # Action: vehicle velocity x,y
        self.action_keys = ['vehicle_vx', 'vehicle_vy']

        self.episodes = [] # List of dictionaries (one per file)
        self.indices = []  # List of (episode_idx, step_idx)
        
        csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
        csv_files.sort() # deterministic order

        print(f"Loading {len(csv_files)} csv files...")
        
        all_obs_data = []
        all_action_data = []

        for episode_idx, file_path in enumerate(csv_files):
            # Read CSV files
            df = pd.read_csv(file_path, header=0, names=self.column_names)
            
            # Extract relevant data as numpy float32
            obs = df[self.obs_keys].values.astype(np.float32)
            action = df[self.action_keys].values.astype(np.float32)
            
            self.episodes.append({
                'obs': obs,
                'action': action
            })
            
            # Collect for stats calculation
            all_obs_data.append(obs)
            all_action_data.append(action)

            # Create indices: specific (episode_idx, step_idx) for every available step
            # We construct a sample for every timestep in the episode so that dataloader can use later
            num_steps = len(obs)
            for i in range(num_steps):
                self.indices.append((episode_idx, i))


        # Compute Statistics for normalization
        all_obs_flat = np.concatenate(all_obs_data, axis=0)
        all_action_flat = np.concatenate(all_action_data, axis=0)

        if self.stats is None:
            self.stats = {
                'obs_min': all_obs_flat.min(axis=0), # axis = 0: column-wise stats
                'obs_max': all_obs_flat.max(axis=0),
                'action_min': all_action_flat.min(axis=0),
                'action_max': all_action_flat.max(axis=0)
            }
            
            # Avoid division by zero by filtering out the edge case
            self.stats['obs_max'] = np.where(self.stats['obs_max'] == self.stats['obs_min'], 
                                            self.stats['obs_max'] + 1e-6, self.stats['obs_max'])
            self.stats['action_max'] = np.where(self.stats['action_max'] == self.stats['action_min'], 
                                                self.stats['action_max'] + 1e-6, self.stats['action_max'])

    def normalize(self, data, key_min, key_max):
        # Normalize to [-1, 1], formula: 2 * (x - min) / (max - min) - 1
        return 2 * (data - key_min) / (key_max - key_min) - 1

    def unnormalize(self, data, key_min, key_max):
        # Unnormalize from [-1, 1] back to original, formula: (x + 1) / 2 * (max - min) + min
        return (data + 1) / 2 * (key_max - key_min) + key_min

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_idx, step_idx = self.indices[idx]
        episode = self.episodes[episode_idx]
        
        obs_data = episode['obs']
        action_data = episode['action']
        episode_len = len(obs_data)

        #|o|o|                             observations: 2
        #| |a|a|a|a|a|a|a|a|               actions executed: 8
        #| |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

        # Observation sequence:
        # We need data from (current - obs_horizon + 1) to current
        start = step_idx - self.obs_horizon + 1
        end = step_idx + 1
        
        if start < 0:
            # Repeat the first frame if we are at the start of an episode
            pad_before = -start
            obs_window = obs_data[0:end]
            # Repeat first frame 'pad_before' times
            pad = np.tile(obs_data[0], (pad_before, 1))
            obs_window = np.concatenate([pad, obs_window], axis=0)
        else:
            obs_window = obs_data[start:end]


        # Action sequence:
        # We need data from current to (current + pred_horizon)
        pred_end = step_idx + self.pred_horizon
        
        if pred_end > episode_len:
            # Padding: Repeat the last frame if we go past the end of an episode
            pad_after = pred_end - episode_len
            action_window = action_data[step_idx:]
            # Repeat last frame 'pad_after' times
            pad = np.tile(action_data[-1], (pad_after, 1))
            action_window = np.concatenate([action_window, pad], axis=0)
        else:
            action_window = action_data[step_idx:pred_end]

        # normalization observation and action
        obs_norm = self.normalize(obs_window, self.stats['obs_min'], self.stats['obs_max'])
        action_norm = self.normalize(action_window, self.stats['action_min'], self.stats['action_max'])

        # Return torch tensors
        nsample = {
            'obs': torch.from_numpy(obs_norm),      # (obs_horizon, 6)
            'action': torch.from_numpy(action_norm) # (pred_horizon, 2)
        }
        return nsample

if __name__ == "__main__":
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    
    dataset_dir = "episodes" 
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory '{dataset_dir}' not found. Please create it and add CSV files.")
        exit()

    # Create Dataset
    dataset = VehicleStateDataset(
        dataset_dir=dataset_dir,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    
    # Save training data statistics (min, max) for each dim
    stats = dataset.stats
    print("Dataset stats keys:", stats.keys())

    # Create Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=1, # Set to 0 if debugging on Windows usually helps
        shuffle=True,
        pin_memory=True,
        persistent_workers=True if os.name != 'nt' else False # False on Windows usually avoids errors
    )

    # Visualize data in batch
    print("Fetching a batch...")
    batch = next(iter(dataloader))
    
    # Expected Shapes:
    # Obs: (Batch, Obs_Horizon, Obs_Dim) -> (256, 2, 6)
    # Action: (Batch, Pred_Horizon, Action_Dim) -> (256, 16, 2)
    print("batch['obs'].shape:", batch['obs'].shape) 
    print("batch['action'].shape", batch['action'].shape)