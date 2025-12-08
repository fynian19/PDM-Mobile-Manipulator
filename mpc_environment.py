import pybullet as p
import pybullet_data
import time
import numpy as np
import pinocchio as pin
import cvxpy as cp

from MPC import MPCController


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
p.setJointMotorControlArray(
    robot_id, 
    controlled_joints, 
    p.VELOCITY_CONTROL, 
    forces=np.zeros(len(controlled_joints))
)

# Create a visual target (Red Sphere)
target_pos_vis = [1.0, -1.0, 0.0] # x, y, arm_height
target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
p.createMultiBody(baseVisualShapeIndex=target_visual, basePosition=target_pos_vis)

# ==========================================
# 3. MAIN CONTROL LOOP
# ==========================================

dt = 0.05  # 20 Hz MPC
u_applied = np.zeros(5) # Start with 0 torque

# Define Reference State (Where we want to go)
# [x=2, y=1, theta=0, arm1=0, arm2=0, velocities=0...]
x_ref = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

print("Starting MPC Loop...")
mpc = MPCController(urdf_path, x_ref)
while True:
    start_time = time.time()
    
    # --- A. READ STATE FROM PYBULLET ---
    joint_states = p.getJointStates(robot_id, controlled_joints)
    q_current = np.array([s[0] for s in joint_states])
    v_current = np.array([s[1] for s in joint_states])
    
    # --- B. LINEARIZE DYNAMICS (Pinocchio) ---
    # We pass the 'pin_data' object so it reuses memory
    #Ac, Bc, d = get_affine_dynamics(pin_model, pin_data, q_current, v_current, u_applied)
    
    # --- C. DISCRETIZE ---
    #Ad = np.eye(10) + Ac * dt
    #Bd = Bc * dt
    #dd = d * dt
    
    # --- D. SOLVE MPC ---
    #x_current = np.concatenate([q_current, v_current])
    u_optimal = mpc.get_control_action(q_current, v_current, u_applied)
    
    # --- E. APPLY CONTROL (Torque) ---
    # Clamp safety limits (PyBullet can explode with huge torques)
    u_applied = np.clip(u_optimal, -1000, 1000)
    
    p.setJointMotorControlArray(
        robot_id, 
        controlled_joints, 
        p.TORQUE_CONTROL, 
        forces=u_applied
    )
    
    # --- F. STEP SIMULATION ---
    p.stepSimulation()
    
    # Try to maintain real-time sync (simplistic)
    elapsed = time.time() - start_time
    if elapsed < dt:
        time.sleep(dt - elapsed)