# # this file should subscribe to the topics and takes real-time states as conditioned input
# # during inference stage, notice how do we handle the velocity commands (latency? interval? warmup?)
# # Use multithreaading to avoid latency problem.

import rclpy
from rclpy.node import Node
import torch
import numpy as np
import collections
import os
import threading 
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry

# Import your model architecture
from model.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

factor = 1

class DiffusionInferenceNode(Node):
    def __init__(self):
        super().__init__('diffusion_inference_node')

        self.ckpt_path = "checkpoints/ckpt_epoch_401.pth" 
        self.control_freq = 5 # Hz (Matches 1/dt of your training data)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_inference_steps = 16 
        
        self.load_checkpoint()

        self.data_lock = threading.Lock()
        
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

        self.create_subscription(Odometry, '/obstacle/odometry', self.cb_obstacle, 10)
        self.create_subscription(PoseStamped, '/box/pose', self.cb_box, 10)
        self.create_subscription(PoseStamped, '/vehicle_blue/pose', self.cb_vehicle, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/vehicle_blue/cmd_vel', 10)

        self.timer = self.create_timer(1.0 / self.control_freq, self.control_loop)
        self.get_logger().info(f"Inference Node initialized on {self.device} with DDIM sampling ({self.num_inference_steps} steps).")

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

        # 2. Initialize DDIM Scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100, # Matches the training step count
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            prediction_type='epsilon'
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps) 

        # 3. Prepare Stats
        self.stats_tensor = {}
        for key, value in self.stats.items():
            self.stats_tensor[key] = torch.from_numpy(value).to(self.device, dtype=torch.float32)

    def cb_obstacle(self, msg):
        with self.data_lock:
            self.latest_state['obstacle'] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    def cb_box(self, msg):
        with self.data_lock:
            self.latest_state['box'] = np.array([msg.pose.position.x, msg.pose.position.y])

    def cb_vehicle(self, msg):
            with self.data_lock:
                # [CORRECTED] PoseStamped access: msg.pose.position
                self.latest_state['vehicle'] = np.array([
                    msg.pose.position.x, 
                    msg.pose.position.y
                ])

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

    def control_loop(self):
        
        # 1. Check data availability and update observation
        with self.data_lock:
            if any(v is None for v in self.latest_state.values()):
                self.get_logger().warn("Waiting for topics...", throttle_duration_sec=2.0)
                return

            curr_obs = np.concatenate([
                self.latest_state['obstacle'], 
                self.latest_state['box'], 
                self.latest_state['vehicle'] 
            ])
            self.obs_deque.append(curr_obs)

            # 2. Warmup: Wait until we have obs_horizon (2 steps)
            if len(self.obs_deque) < self.config['obs_horizon']:
                self.get_logger().warn("Warming up observation buffer...", throttle_duration_sec=0.5)
                return

        # 3. Execution Logic (Receding Horizon)
        # If the action queue is empty, calculate a new plan
        if len(self.action_queue) == 0:
            self.run_inference()
        
        # 4. Pop and publish next action
        action_to_pub = None
        with self.data_lock:
            if len(self.action_queue) > 0:
                action_to_pub = self.action_queue.popleft()
        
        if action_to_pub is not None:
            self.publish_cmd_vel(action_to_pub)
        else:
            # Safety stop if the inference step takes too long or failed
            self.cmd_vel_pub.publish(Twist())


    def run_inference(self):
        """ Runs the DDIM reverse process to generate a new action chunk. """
        
        # 1. Prepare Observation Batch
        with self.data_lock:
            obs_seq = np.stack(self.obs_deque)
        
        obs_tensor = torch.from_numpy(obs_seq).to(self.device, dtype=torch.float32)
        nobs = self.normalize_obs(obs_tensor)
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

        # 2. DDIM Reverse Process 
        with torch.no_grad():
            B = 1
            # Start from Gaussian Noise
            naction = torch.randn(
                (B, self.config['pred_horizon'], self.config['action_dim']), 
                device=self.device
            )
            
            # DDIM Denoising loop (using only 16 steps)
            for k in self.noise_scheduler.timesteps: 
                # Predict noise using the trained model
                noise_pred = self.model(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # Inverse step using DDIM formula
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # 3. Unnormalize and Store
        naction = naction.detach()[0] 
        action_pred = self.unnormalize_action(naction) 
        action_pred_np = action_pred.cpu().numpy()

        # 4. Fill Action Queue (Chunking - Thread Safe)
        with self.data_lock:
            # Clear old actions (Receding Horizon)
            self.action_queue.clear() 
            # Only take action_horizon steps
            valid_actions = action_pred_np[:self.config['action_horizon']]
            
            for act in valid_actions:
                self.action_queue.append(act)
        
        self.get_logger().info(f"DDIM inference complete in {self.num_inference_steps} steps. New action chunk size: {len(valid_actions)}", throttle_duration_sec=1.0)


    def publish_cmd_vel(self, action):
        # action is [vx, vy] 
        msg = Twist()
        
        # Velocity Scaling / Safety Clipping
        vx = np.clip(action[0], -1000.0, 10000.0) 
        vy = np.clip(action[1], -1000.0, 1000.0)

        msg.linear.x = float(vx) / factor
        msg.linear.y = float(vy) / factor
        msg.angular.z = 0.0 
        self.get_logger().info(f"Publishing Cmd: vx={vx:.3f}, vy={vy:.3f}, Queue Len={len(self.action_queue)}", throttle_duration_sec=0.1)
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

