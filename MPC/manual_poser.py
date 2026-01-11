import pybullet as p
import pybullet_data
import time
import numpy as np
from environment_loader import load_environment_from_txt

# ==========================================
# SETUP
# ==========================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Load Robot
robot_id = p.loadURDF("URDF/mobileManipulator.urdf", useFixedBase=True)

# Load Obstacles (Visual Reference)
load_environment_from_txt("scenario_6_obstacles.txt")

# Create Sliders for every joint
joint_ids = []
param_ids = []
joint_names = []

# 1. Base Sliders (Since it's fixed base sim)
# We will manually move the base using resetBasePosition
base_sliders = {
    "x": p.addUserDebugParameter("Base X", -10, 10, 0.0),
    "y": p.addUserDebugParameter("Base Y", -10, 10, 7.6), # Start near goal
    "th": p.addUserDebugParameter("Base Theta", -3.14, 3.14, 0.0) # Face goal
}

# 2. Arm Sliders
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    name = info[1].decode()
    # Skip wheels if they are controlled by base sliders logic
    if "mobile" in name: continue 
    
    # Add slider for arm joints
    lower, upper = info[8], info[9]
    if lower > upper: # Unlimited joint
        lower, upper = -3.14, 3.14
        
    uid = p.addUserDebugParameter(name, lower, upper, 0.0)
    param_ids.append(uid)
    joint_ids.append(i)
    joint_names.append(name)

print("------------------------------------------------")
print("ADJUST SLIDERS UNTIL ROBOT TOUCHES BOX.")
print("The code below will print the configuration.")
print("------------------------------------------------")

while True:
    # 1. Update Base
    x = p.readUserDebugParameter(base_sliders["x"])
    y = p.readUserDebugParameter(base_sliders["y"])
    th = p.readUserDebugParameter(base_sliders["th"])
    p.resetBasePositionAndOrientation(robot_id, [x, y, 0.02], p.getQuaternionFromEuler([0,0,th]))
    
    # 2. Update Arm
    q_arm = []
    for i, param_id in enumerate(param_ids):
        angle = p.readUserDebugParameter(param_id)
        p.resetJointState(robot_id, joint_ids[i], angle)
        q_arm.append(angle)
    
    # 3. Print the Magic Vector
    # Format: [BaseX, BaseY, Theta, J1, J2...]
    full_config = [x, y, th] + q_arm
    
    # Simple formatted print
    print("\n\nCOPY THIS ARRAY INTO YOUR CODE:")
    print(f"HARDCODED_TARGET_Q = np.array({str(full_config)})")
    
    time.sleep(0.1)