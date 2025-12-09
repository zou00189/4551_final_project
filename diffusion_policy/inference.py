# # this file should subscribe to the topics and takes real-time states as conditioned input
# # during inference stage, notice how do we handle the velocity commands (latency? interval? warmup?)
# # Use multithreaading to avoid latency problem.

import rclpy
from rclpy.node import Node
import torch
import numpy as np
import collections
import os
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry

# Import your model architecture
# Ensure model/conditional_unet1d.py is in your python path
from model.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class DiffusionInferenceNode(Node):
    def __init__(self):
        super().__init__('diffusion_inference_node')

        # --- 1. Settings ---
        self.ckpt_path = "checkpoints/ckpt_epoch_151.pth" # Update this to your best epoch
        self.control_freq = 30.0 # Hz (Matches 1/dt of your training data)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        
        # --- 2. Load Model & Stats ---
        self.load_checkpoint()

        # --- 3. Runtime State ---
        # Stores latest raw messages
        self.latest_state = {
            'obstacle': None, # [x, y]
            'box': None,      # [x, y]
            'vehicle': None   # [x, y]
        }
        
        # Observation History (Matches obs_horizon=2)
        self.obs_deque = collections.deque(maxlen=self.config['obs_horizon'])
        
        # Action Queue (Matches action_horizon=8)
        self.action_queue = collections.deque(maxlen=self.config['action_horizon'])

        # --- 4. ROS Subscribers ---
        # Obstacle (Odometry)
        self.create_subscription(Odometry, '/obstacle/odometry', self.cb_obstacle, 10)
        # Box (PoseStamped)
        self.create_subscription(PoseStamped, '/box/pose', self.cb_box, 10)
        # Vehicle (Odometry)
        self.create_subscription(Odometry, '/vehicle_blue/odometry', self.cb_vehicle, 10)

        # --- 5. Publisher ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/vehicle_blue/cmd_vel', 10)

        # --- 6. Timer Loop ---
        self.timer = self.create_timer(1.0 / self.control_freq, self.control_loop)
        self.get_logger().info(f"Inference Node initialized on {self.device}")

    def load_checkpoint(self):
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")
            
        payload = torch.load(self.ckpt_path, map_location=self.device)
        self.config = payload['config']
        self.stats = payload['stats']
        
        self.get_logger().info(f"Loaded Config: {self.config}")

        # 1. Initialize Model
        self.model = ConditionalUnet1D(
            input_dim=self.config['action_dim'],
            global_cond_dim=self.config['obs_dim'] * self.config['obs_horizon']
        ).to(self.device)
        
        self.model.load_state_dict(payload['model_state_dict'])
        self.model.eval()

        # 2. Initialize Scheduler (MUST MATCH TRAINING)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            prediction_type='epsilon'
        )

        # 3. Prepare Stats as Tensors for fast GPU normalization
        self.stats_tensor = {}
        for key, value in self.stats.items():
            self.stats_tensor[key] = torch.from_numpy(value).to(self.device, dtype=torch.float32)

    # --- Callbacks ---
    def cb_obstacle(self, msg):
        self.latest_state['obstacle'] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    def cb_box(self, msg):
        self.latest_state['box'] = np.array([msg.pose.position.x, msg.pose.position.y])

    def cb_vehicle(self, msg):
        self.latest_state['vehicle'] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    # --- Normalization Helpers ---
    def normalize_obs(self, obs_tensor):
        # Formula: 2 * (x - min) / (max - min) - 1
        k_min = self.stats_tensor['obs_min']
        k_max = self.stats_tensor['obs_max']
        return 2 * (obs_tensor - k_min) / (k_max - k_min) - 1

    def unnormalize_action(self, action_tensor):
        # Formula: (x + 1) / 2 * (max - min) + min
        k_min = self.stats_tensor['action_min']
        k_max = self.stats_tensor['action_max']
        return (action_tensor + 1) / 2 * (k_max - k_min) + k_min

    # --- Main Control Loop ---
    def control_loop(self):
        # 1. Check data availability
        if any(v is None for v in self.latest_state.values()):
            self.get_logger().warn("Waiting for topics...", throttle_duration_sec=2.0)
            return

        # 2. Construct Observation Vector
        # CRITICAL: Must match self.obs_keys = ['obstacle_x', 'obstacle_y', 'box_x', 'box_y', 'vehicle_x', 'vehicle_y']
        curr_obs = np.concatenate([
            self.latest_state['obstacle'], # 2
            self.latest_state['box'],      # 2
            self.latest_state['vehicle']   # 2
        ]) # Total 6 dims

        # 3. Add to deque
        self.obs_deque.append(curr_obs)

        # 4. Warmup: Wait until we have obs_horizon (2 steps)
        if len(self.obs_deque) < self.config['obs_horizon']:
            return

        # 5. Execution Logic
        if len(self.action_queue) == 0:
            # Queue empty: Plan new trajectory
            self.run_inference()
        
        # Pop and publish next action
        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            self.publish_cmd_vel(action)

    def run_inference(self):
        # 1. Prepare Observation Batch
        # Stack deque -> (obs_horizon, obs_dim) -> (2, 6)
        obs_seq = np.stack(self.obs_deque)
        obs_tensor = torch.from_numpy(obs_seq).to(self.device, dtype=torch.float32)
        
        # Normalize
        nobs = self.normalize_obs(obs_tensor)
        
        # Flatten and Add Batch Dim: (1, obs_horizon * obs_dim) -> (1, 12)
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

        # 2. Diffusion Reverse Process
        with torch.no_grad():
            B = 1
            # Start from Gaussian Noise
            naction = torch.randn(
                (B, self.config['pred_horizon'], self.config['action_dim']), 
                device=self.device
            )
            
            # Init scheduler
            self.noise_scheduler.set_timesteps(100)

            # Denoising loop
            for k in self.noise_scheduler.timesteps:
                # Predict noise
                noise_pred = self.model(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # Inverse step
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # 3. Unnormalize and Store
        naction = naction.detach()[0] # Remove Batch dim
        action_pred = self.unnormalize_action(naction) # (pred_horizon, 2)
        
        # Move to CPU numpy
        action_pred_np = action_pred.cpu().numpy()

        # 4. Fill Action Queue (Chunking)
        # Only take action_horizon steps (0 to 8)
        valid_actions = action_pred_np[:self.config['action_horizon']]
        
        for act in valid_actions:
            self.action_queue.append(act)

    def publish_cmd_vel(self, action):
        # action is [vx, vy] (Global or Robot frame depends on training data)
        msg = Twist()
        
        # Velocity Scaling / Safety Clipping
        # Example: Clip to max 1.0 m/s
        vx = np.clip(action[0], -1000.0, 10000.0)  
        vy = np.clip(action[1], -1000.0, 1000.0)

        # NOTE on Coordinate Frames:
        # If your training data 'vehicle_vx' is in GLOBAL frame, but the robot
        # expects BODY frame commands, you must rotate this vector by the 
        # inverse of the robot's current yaw.
        # Assuming training data was recorded relative to robot or robot is holonomic/aligned:
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = 0.0 
        self.get_logger().info(f"Publishing Cmd: vx={vx:.3f}, vy={vy:.3f}", throttle_duration_sec=0.5)
        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DiffusionInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot on exit
        stop_msg = Twist()
        node.cmd_vel_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

