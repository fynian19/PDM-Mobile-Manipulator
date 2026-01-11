import pybullet as p
import pybullet_data
import time
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

# Custom Imports
from MPC import MPCController, MPCVisualizer, get_clamped_reference, update_c_space_distance, update_ee_distance
from environment_loader import load_environment_from_txt

# ==========================================
# 1. PYBULLET SETUP
# ==========================================
urdf_path = "URDF/mobileManipulator.urdf"

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Improve Camera & Mouse
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=135, cameraPitch=-35, cameraTargetPosition=[0,0,0])

robot_id = p.loadURDF(urdf_path, useFixedBase=True)

# ==========================================
# 2. LOAD OBSTACLES (CLEAN)
# ==========================================
obs_ids, obs_data_list = load_environment_from_txt("scenario_6.5_obstacles.txt")

mpc_obstacle_list = []

print(f"--> Filtering {len(obs_data_list)} objects for MPC...")

for obs in obs_data_list:
    # If the loader flagged it as a ghost (The Top Box), SKIP IT for MPC
    if obs.get('is_ghost', False):
        print(f"   [IGNORED] Skipping Ghost Object at {obs['pos']} for MPC.")
        continue
    
    # Otherwise, it is a physical obstacle (Walls, Cylinders, Pedestal)
    mpc_obstacle_list.append(obs)

print(f"--> MPC configured with {len(mpc_obstacle_list)} physical obstacles.")

# ==========================================
# 3. ROBOT CONFIGURATION
# ==========================================
controlled_joints = [0, 1, 2, 3, 4] 

# Start Position (Outside the room)
#start_pos = [-9.5, -7.0, 0]    # Original
start_pos = [-9, -9, 0] 
p.resetJointState(robot_id, 0, start_pos[0])
p.resetJointState(robot_id, 1, start_pos[1])
p.resetJointState(robot_id, 2, start_pos[2])

# Disable Motors for Torque Control
p.setJointMotorControlArray(robot_id, controlled_joints, p.VELOCITY_CONTROL, forces=[0]*5)

# ==========================================
# 4. MPC & CONTROLLER SETUP
# ==========================================
dt = 0.02
no_steps = 15

# Global Targets
TARGET_BOX_XY = np.array([0.0, 7.0])
HARDCODED_X = np.array([0.0, 7, 1.57, 1.6, 1.6] + [0]*5)
PARKING_X = np.array([0, 7, 0.0, 0.0, 0.0] + [0]*5) 

# Mode Weights
WEIGHTS_NAV = np.array([2000, 2000, 0.01, 0.01, 0.01] + [10, 10, 1, 1, 1])
WEIGHTS_MANIP = np.array([2000, 2000, 2000, 2000, 2000] + [10, 10, 10, 10, 10])

# --- INIT MPC WITH OBSTACLES ---
wall_bounds = {'x_min': -10.0, 'x_max': 10.0, 'y_min': -10.0, 'y_max': 10.0}

mpc = MPCController(
    urdf_path, 
    PARKING_X, 
    dt*no_steps, 
    N=40, 
    obstacle_list=mpc_obstacle_list,  # <--- PASS THE FILTERED LIST
    bounds=wall_bounds
)

viz = MPCVisualizer(p)

# --- SETUP TRACKING (EE + JOINTS) ---
# 1. End Effector Setup
total_ee_dist = 0.0
prev_ee_pos = None
ee_link_idx = p.getNumJoints(robot_id) - 1 

# 2. C-Space (Joint) Setup
total_joint_dists = np.zeros(5) # [BaseX, BaseY, Theta, Shoulder, Elbow]
prev_q = None
# ==========================================
# 5. MAIN LOOP
# ==========================================
u_applied = np.zeros(5)
pedestal_switch = False
CURRENT_MODE = "INIT"
log_t, log_q, log_v, log_u, log_ref = [], [], [], [], []
sim_start_time = time.time()

print("Starting Loop...")
print("\n=== JOINT MAPPING VERIFICATION ===")
for i, joint_idx in enumerate(controlled_joints):
    # index 1 is the joint name
    name = p.getJointInfo(robot_id, joint_idx)[1].decode("utf-8") 
    print(f"Index {i} -> Joint ID {joint_idx}: {name}")
print("==================================\n")
try:
    while True:
        start_time = time.time()
        

        # 1. State Estimation
        states = p.getJointStates(robot_id, controlled_joints)
        q_curr = np.array([s[0] for s in states])
        v_curr = np.array([s[1] for s in states])
        x_curr_vec = np.concatenate([q_curr, v_curr])
        # 1. Read State
        states = p.getJointStates(robot_id, controlled_joints)
        q_curr = np.array([s[0] for s in states])
        
                #--- TRACKING UPDATES ---
        # A. End Effector
        prev_ee_pos, total_ee_dist = update_ee_distance(p, robot_id, ee_link_idx, prev_ee_pos, total_ee_dist)
        
        # B. Configuration Space (Joints)
        prev_q, total_joint_dists = update_c_space_distance(q_curr, prev_q, total_joint_dists)

        # --- PERIODIC PRINTING (Every 50 steps) ---
        if len(log_t) % 5 == 0:
            print(f"--- Step {len(log_t)} ---")
            print(f"--- Time {time.time() - sim_start_time} ---")
            print(f"EE Path:    {total_ee_dist:.3f} m")
            # Format array nicely for printing
            j_dists_str = np.array2string(total_joint_dists, precision=3, separator=', ')
            print(f"Joint Path: {j_dists_str}")


        # --- CAMERA UPDATE ---
        #p.resetDebugVisualizerCamera(cameraDistance=6.0, cameraYaw=300,cameraPitch=-60, cameraTargetPosition=[q_curr[0], q_curr[1], 0.0] )

        # YAW FIXED CAMERA
        #yaw_degrees = np.degrees(q_curr[2]) - 90 
        #p.resetDebugVisualizerCamera(cameraDistance=6.0, cameraYaw=yaw_degrees +45, cameraPitch=-60,cameraTargetPosition=[q_curr[0], q_curr[1], 0.5])

        # 2. Logic & Mode Switching
        dist_to_target = np.linalg.norm(q_curr[:2] - TARGET_BOX_XY)
        
        if dist_to_target > 0.5:
            if CURRENT_MODE != "NAV":
                print(f"Dist {dist_to_target:.2f}m -> NAV MODE")
                mpc.update_weights(WEIGHTS_NAV)
                CURRENT_MODE = "NAV"
            # Carrot Following
            x_ref_local = get_clamped_reference(x_curr_vec, PARKING_X, max_lookahead=2.0)
            mpc.x_ref_val = x_ref_local
        else:
            if CURRENT_MODE != "MANIP":
                print(f"Dist {dist_to_target:.2f}m -> MANIP MODE")
                mpc.update_weights(WEIGHTS_MANIP)
                CURRENT_MODE = "MANIP"
            mpc.x_ref_val = HARDCODED_X
            x_ref_local = HARDCODED_X

   
# --- MEASURE MPC COMPUTATION TIME ---
        t0_mpc = time.perf_counter()
        
        u_optimal, X_pred, vis_data, pedestal_switch = mpc.get_control_action(q_curr, v_curr, u_applied, pedestal_switch)
        
        t1_mpc = time.perf_counter()
        mpc_duration = t1_mpc - t0_mpc
        
        # 4. Visualization
        viz.draw_trajectory(X_pred)
        # This will now work because mpc.obstacle_list is not empty!
        viz.draw_planes(vis_data) 
        
        # 5. Apply Control
        if u_optimal is not None:
            u_applied = np.clip(u_optimal, -300, 300)
            for _ in range(no_steps): 
                p.setJointMotorControlArray(robot_id, controlled_joints, p.TORQUE_CONTROL, forces=u_applied)
                p.stepSimulation()
        else:
            # Emergency Stop if MPC fails
            p.setJointMotorControlArray(robot_id, controlled_joints, p.VELOCITY_CONTROL, forces=[0]*5)
            p.stepSimulation()
        

        # --- MEASURE SIMULATION STEP TIME ---
        t0_sim = time.perf_counter()
        
        if u_optimal is not None:
            u_applied = np.clip(u_optimal, -300, 300)
            for _ in range(no_steps): 
                p.setJointMotorControlArray(robot_id, controlled_joints, p.TORQUE_CONTROL, forces=u_applied)
                p.stepSimulation()
        
        t1_sim = time.perf_counter()
        sim_duration = t1_sim - t0_sim

        # --- PRINT METRICS ---
        # Print only occasionally to avoid slowing down the console
        if len(log_t) % 10 == 0:
            print(f"Step {len(log_t)} | MPC Calc: {mpc_duration*1000:.1f} ms | Physics: {sim_duration*1000:.1f} ms")

        # 6. Logging
        log_t.append(time.time() - sim_start_time)
        log_q.append(q_curr); log_v.append(v_curr); log_u.append(u_applied); log_ref.append(x_ref_local)

        elapsed = time.time() - start_time
        if elapsed < dt: time.sleep(dt - elapsed)

except KeyboardInterrupt:
    print("\nStopped.")
print(f"FINAL METRIC: The End-Effector traced a total of {total_ee_dist:.4f} meters.")

# ==========================================
# 7. C-SPACE PATH LENGTH ANALYSIS
# ==========================================
print("\n" + "="*50)
print(" CONFIGURATION SPACE PATH TRACED (PER JOINT)")
print("="*50)

# 1. Calculate the step-by-step difference
# q_arr has shape (TimeSteps, 5)
# diffs has shape (TimeSteps-1, 5)
diffs = np.diff(q_arr, axis=0)

# 2. Take the absolute value (treat backward movement as positive distance)
abs_diffs = np.abs(diffs)

# 3. Sum up all the small steps
path_lengths = np.sum(abs_diffs, axis=0)

# 4. Display
joint_names = ["Base X", "Base Y", "Base Theta", "Shoulder", "Elbow"]
units       = ["m", "m", "rad", "rad", "rad"]

print(f"{'JOINT':<15} | {'PATH TRACED':<15} | {'DISPLACEMENT':<15} | {'UNIT'}")
print("-" * 65)

for i in range(len(joint_names)):
    # Path Traced: Sum of all movements
    traced = path_lengths[i]
    
    # Displacement: Just Final - Initial (for comparison)
    disp = q_arr[-1, i] - q_arr[0, i]
    
    print(f"{joint_names[i]:<15} | {traced:10.4f}      | {disp:10.4f}      | {units[i]}")

print("-" * 65)
print("Note: 'Path Traced' includes back-and-forth motion.")
print("      'Displacement' is just Final Position - Start Position.")

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
