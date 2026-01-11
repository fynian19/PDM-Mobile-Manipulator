import pybullet as p
import pybullet_data
import time

# --- YOUR ROUGH IDEA GOES HERE (In Radians) ---
# Example: Shoulder 0 degrees, Elbow -90 degrees (-1.57 rad)
MY_ARM_ANGLES = [0.6, 0.6] 

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("URDF/mobileManipulator.urdf", useFixedBase=True)

# 1. Set Base to Parking Spot
p.resetJointState(robot_id, 0, 0.0)  # X
p.resetJointState(robot_id, 1, 7.2)  # Y
p.resetJointState(robot_id, 2, 1.57) # Theta

# 2. Set Arm Joints (Indices 3, 4...)
for i, angle in enumerate(MY_ARM_ANGLES):
    p.resetJointState(robot_id, 3 + i, angle)

print(f"Showing config: Base=[0, 7.2, 1.57], Arm={MY_ARM_ANGLES}")
print("If this looks good, copy these values into MPC.py!")

while True:
    p.stepSimulation()
    time.sleep(0.01)