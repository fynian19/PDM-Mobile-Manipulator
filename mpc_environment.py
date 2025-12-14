import pybullet as p
import pybullet_data
import time
import numpy as np
import pinocchio as pin
import cvxpy as cp
import matplotlib.pyplot as plt
from MPC import MPCController, MPCVisualizer, get_clamped_reference


# Setup Pinocchio Model ONCE
urdf_path = "URDF/mobileManipulator.urdf" # Ensure this path is correct
pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()

# Connect PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81) # Gravity is important!
p.loadURDF("plane.urdf")

# Load Robot
robot_id = p.loadURDF(urdf_path, useFixedBase=True) # It acts fixed because of the Prismatic joints

# Setup Joints
joint_map = {}
controlled_joints = []
for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    name = info[1].decode()
    joint_map[name] = i
    # We want to control the prismatic base + arm joints
    if name in ["joint_mobile_x", "joint_mobile_y", "joint_mobile_theta", 
                "joint_base_to_upper_arm", "joint_upper_to_lower_arm"]:
        controlled_joints.append(i)

# IMPORTANT: Turn off PyBullet's default velocity motors to use Torque Control
p.setJointMotorControlArray(robot_id, controlled_joints, p.VELOCITY_CONTROL, forces=np.zeros(len(controlled_joints)))

## ==========================================
# MAIN CONTROL LOOP
# ==========================================

# Sim setup
dt = 0.02  # PyBullet physics time step
no_steps = 15 # Number of physics steps per MPC step

# Define Reference State (Where we want to go)
x_ref = np.array([30, 7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Create a visual target (Red Sphere)
target_pos_vis = x_ref[:3] # x, y, arm_height
target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
p.createMultiBody(baseVisualShapeIndex=target_visual, basePosition=target_pos_vis)

mpc = MPCController(urdf_path, x_ref, dt*no_steps)
viz = MPCVisualizer(p)

u_applied = np.zeros(5) # Start with 0 torque

# ==========================================
# LOGGING INITIALIZATION
# ==========================================
# We create lists to store the data history
log_t = []
log_q = [] # Joint Positions
log_v = [] # Joint Velocities
log_u = [] # Control Inputs
log_ref = [] # Reference history

print("Starting MPC Loop...")
print("Press Ctrl+C to stop simulation and show plots.")

sim_start_time = time.time()
try:
    while True:
        start_time = time.time()
        
        # --- A. READ STATE FROM PYBULLET ---
        joint_states = p.getJointStates(robot_id, controlled_joints)
        q_current = np.array([s[0] for s in joint_states])
        v_current = np.array([s[1] for s in joint_states])
        # Combine into state vector x
        x_curr_vec = np.concatenate([q_current, v_current])
        print(f"Current state: {x_curr_vec}")

        # --- B. LOG DATA ---
        current_time = start_time - sim_start_time
        log_t.append(current_time)
        log_q.append(q_current)
        log_v.append(v_current)
        log_u.append(u_applied) # Log the PREVIOUS applied torque (or current if you prefer)
        #log_ref.append(x_ref)
        
        # --- D. SOLVE MPC ---
        #print(mpc.get_control_action(q_current, v_current, u_applied))
        x_ref_local = get_clamped_reference(x_curr_vec, x_ref, max_lookahead=3)
        
        # Update MPC with this new local target
        mpc.x_ref_val = x_ref_local

        u_optimal, X_pred = mpc.get_control_action(q_current, v_current, u_applied)
        log_ref.append(x_ref_local)
        # --- D. VISUALIZE MPC TRAJECTORY ---
        viz.draw_trajectory(X_pred)

        print(f"Optimal Control: {u_optimal}")
        # --- E. APPLY CONTROL (Torque) ---
        # Clamp safety limits (PyBullet can explode with huge torques)
        u_applied = np.clip(u_optimal, -500, 500)
        
        # Step simulation multiple times
        for _ in range(no_steps): 
            p.setJointMotorControlArray(robot_id, controlled_joints, p.TORQUE_CONTROL, forces=u_applied)
            p.stepSimulation()
        
        # --- G. REAL-TIME SYNC ---
        # Now we sleep only if the calculation + physics was faster than 0.05s
        elapsed = time.time() - start_time
        if elapsed < dt:
            time.sleep(dt - elapsed)

except KeyboardInterrupt:
    print("\nSimulation Interrupted by User.")


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