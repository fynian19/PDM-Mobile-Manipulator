import pybullet as p
import pybullet_data
import time
import numpy as np
import pinocchio as pin
import cvxpy as cp
import matplotlib.pyplot as plt

from MPC import MPCController, MPCVisualizer, get_clamped_reference
from environment_loader import load_environment_from_txt

# ==========================================
# SETUP
# ==========================================
urdf_path = "URDF/mobileManipulator.urdf"
pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Enable mouse interaction and free look
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

robot_id = p.loadURDF(urdf_path, useFixedBase=True)

# --- LOAD OBSTACLES (Correct Way) ---
# This function now handles BOX and CYL and gives us the math data
obs_ids, obs_data_list = load_environment_from_txt("scenario_6_obstacles.txt")
print(f"Loaded {len(obs_ids)} obstacles.")

# Setup Joints
controlled_joints = []
joint_map = {}
for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    name = info[1].decode()
    joint_map[name] = i
    if name in ["joint_mobile_x", "joint_mobile_y", "joint_mobile_theta", 
                "joint_base_to_upper_arm", "joint_upper_to_lower_arm"]:
        controlled_joints.append(i)

# --- START POSITION (Outside the room) ---
start_pos = [-8.5, -7, 1.57] 
#start_pos = [0, 0, 0] 

p.resetJointState(robot_id, joint_map["joint_mobile_x"], start_pos[0])
p.resetJointState(robot_id, joint_map["joint_mobile_y"], start_pos[1])
p.resetJointState(robot_id, joint_map["joint_mobile_theta"], start_pos[2])

p.setJointMotorControlArray(robot_id, controlled_joints, p.VELOCITY_CONTROL, forces=np.zeros(len(controlled_joints)))

# ==========================================
# MPC INIT
# ==========================================
dt = 0.02
no_steps = 15
x_ref = np.array([0, 8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Goal at (0, 8)

# Pass the LIST of obstacles
wall_bounds = {'x_min': -10.0, 'x_max': 10.0, 'y_min': -10.0, 'y_max': 10.0}

mpc = MPCController(urdf_path, x_ref, dt*no_steps, N=50, 
                    obstacle_list=obs_data_list, 
                    bounds=wall_bounds)
viz = MPCVisualizer(p)


u_applied = np.zeros(5)
target_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1])
p.createMultiBody(baseVisualShapeIndex=target_vis, basePosition=x_ref[:3])

# ... (Rest of loop matches your previous code) ...
print("Starting MPC Loop...")
log_t, log_q, log_v, log_u, log_ref = [], [], [], [], []
sim_start_time = time.time()

try:
    while True:
        start_time = time.time()
        
        # 1. Read
        joint_states = p.getJointStates(robot_id, controlled_joints)
        q_curr = np.array([s[0] for s in joint_states])
        v_curr = np.array([s[1] for s in joint_states])
        x_curr_vec = np.concatenate([q_curr, v_curr])
        
        # 2. Log
        log_t.append(start_time - sim_start_time)
        log_q.append(q_curr); log_v.append(v_curr); log_u.append(u_applied)

        # 3. Ref (Lookahead 2.0 works well for corners)
        x_ref_local = get_clamped_reference(x_curr_vec, x_ref, max_lookahead=2.0)
        mpc.x_ref_val = x_ref_local
        log_ref.append(x_ref_local)

        # 4. MPC
        u_optimal, X_pred, vis_data = mpc.get_control_action(q_curr, v_curr, u_applied)
        

        # 5. Vis
        viz.draw_trajectory(X_pred)
        viz.draw_planes(vis_data)
        
        # 6. Apply
        u_applied = np.clip(u_optimal, -500, 500)
        for _ in range(no_steps): 
            p.setJointMotorControlArray(robot_id, controlled_joints, p.TORQUE_CONTROL, forces=u_applied)
            p.stepSimulation()
        
        elapsed = time.time() - start_time
        if elapsed < dt: time.sleep(dt - elapsed)

except KeyboardInterrupt:
    print("\nStopped.")
    
# ... (Use the Safety Trim Plotting Code from before) ...


# ==========================================
# PLOTTING
# ==========================================
print("Generating Plots...")

# Convert lists to numpy arrays
t_arr = np.array(log_t)
q_arr = np.array(log_q)
v_arr = np.array(log_v)
u_arr = np.array(log_u)
ref_arr = np.array(log_ref)

# Define Joint Groups
base_indices = [0, 1, 2]
base_names = ["Base X", "Base Y", "Base Theta"]

arm_indices = [3, 4]
arm_names = ["Shoulder", "Elbow"]

# ==========================================
# FIGURE 1: BASE DASHBOARD (Map + Theta + Effort)
# ==========================================
fig1, ax1 = plt.subplots(2, 1, figsize=(8, 14)) # 3 Rows now

# --- Subplot 1: 2D Spatial Path (X vs Y) with Orientation Arrows ---
ax1[0].set_title("Base Navigation: Path & Heading")
ax1[0].plot(q_arr[:, 0], q_arr[:, 1], label="Robot Path", linewidth=3, color='blue', alpha=0.6)

# Draw Heading Arrows (Quiver) - Only draw every 10th step to avoid clutter
step = max(1, len(t_arr) // 20) # Draw ~20 arrows total
ax1[0].quiver(
    q_arr[::step, 0], q_arr[::step, 1],  # X, Y positions
    np.cos(q_arr[::step, 2]), np.sin(q_arr[::step, 2]), # Direction (Cos, Sin)
    color='black', scale=20, width=0.005, headwidth=4, label="Heading"
)

# Start/End Markers
ax1[0].scatter(q_arr[0, 0], q_arr[0, 1], color='green', label="Start", zorder=5, s=100)
ax1[0].scatter(ref_arr[-1, 0], ref_arr[-1, 1], color='red', label="Target", zorder=5, s=100)

ax1[0].set_xlabel("Base X Position [m]")
ax1[0].set_ylabel("Base Y Position [m]")
ax1[0].axis('equal') 
ax1[0].grid(True)
ax1[0].legend()

# --- Subplot 2: Base Control Effort ---
for i, idx in enumerate(base_indices):
    color = plt.cm.tab10(i)
    ax1[1].plot(t_arr, u_arr[:, idx], label=f"{base_names[i]} Effort", color=color)

ax1[1].set_title("Base Control Effort (Forces & Torques)")
ax1[1].set_ylabel("Force [N] / Torque [Nm]")
ax1[1].set_xlabel("Time [s]")
ax1[1].legend(loc='upper right')
ax1[1].grid(True)

plt.tight_layout()


# ==========================================
# FIGURE 2: ARM PARAMETERS (Shoulder + Elbow)
# ==========================================
fig2, ax2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# --- 1. Arm Positions ---
for i, idx in enumerate(arm_indices):
    color = plt.cm.tab10(i + 3) # Shift colors
    ax2[0].plot(t_arr, q_arr[:, idx], label=f"{arm_names[i]}", color=color, linewidth=2)
    ax2[0].plot(t_arr, ref_arr[:, idx], linestyle='--', color=color, alpha=0.5)

ax2[0].set_title("Arm Joint Positions")
ax2[0].set_ylabel("Angle [rad]")
ax2[0].legend(loc='upper right')
ax2[0].grid(True)

# --- 2. Arm Velocities ---
for i, idx in enumerate(arm_indices):
    color = plt.cm.tab10(i + 3)
    ax2[1].plot(t_arr, v_arr[:, idx], label=f"{arm_names[i]} Vel", color=color)

ax2[1].set_title("Arm Joint Velocities")
ax2[1].set_ylabel("Velocity [rad/s]")
ax2[1].grid(True)

# --- 3. Arm Torques ---
for i, idx in enumerate(arm_indices):
    color = plt.cm.tab10(i + 3)
    ax2[2].plot(t_arr, u_arr[:, idx], label=f"{arm_names[i]} Torque", color=color)

ax2[2].set_title("Arm Control Effort")
ax2[2].set_ylabel("Torque [Nm]")
ax2[2].set_xlabel("Time [s]")
ax2[2].grid(True)

plt.tight_layout()
plt.show()
