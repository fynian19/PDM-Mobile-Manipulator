import pybullet as p
import pybullet_data
import time
import math

from environment_loader import load_environment_from_txt


# ======================================================================
# PYBULLET INITIALIZATION
# ======================================================================

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.setGravity(0, 0, 0)

p.loadURDF("plane.urdf")
robot = p.loadURDF("URDF/mobileManipulator.urdf")


# ======================================================================
# LOAD ENVIRONMENT
# ======================================================================

obstacle_ids, laser_ids, laser_base_positions = load_environment_from_txt(
    "scenario_7_obstacles.txt"
)

print(f"Loaded {len(obstacle_ids)} static obstacles.")
print(f"Loaded {len(laser_ids)} dynamic lasers.")


# ======================================================================
# JOINT MAP
# ======================================================================

joint_map = {}
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    name = info[1].decode()
    joint_map[name] = i

print("Joint map:", joint_map)

# Enable motors for all joints
for j in range(p.getNumJoints(robot)):
    p.setJointMotorControl2(
        robot,
        j,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=100
    )

# Arm joints
arm_joints = [
    joint_map["joint_base_to_upper_arm"],
    joint_map["joint_upper_to_lower_arm"]
]


# ======================================================================
# LASER MOTION PARAMETERS
# ======================================================================

LASER_MIN_Z = 0.0
LASER_MAX_Z = 2.2
LASER_SPEED = 0.02  # radians per step


# ======================================================================
# MAIN SIMULATION LOOP
# ======================================================================

for step in range(800):

    # --------------------------------------------------
    # Animate lasers (alternating motion)
    # --------------------------------------------------
    for i, laser_id in enumerate(laser_ids):
        base_x, base_y, base_z = laser_base_positions[i]

        # Decide phase based on starting height
        if abs(base_z - LASER_MIN_Z) < abs(base_z - LASER_MAX_Z):
            phase = 0.0          # start going UP
        else:
            phase = math.pi     # start going DOWN

        z_range = LASER_MAX_Z - LASER_MIN_Z
        z_offset = 0.5 * z_range * (1 + math.sin(step * LASER_SPEED + phase))
        new_z = LASER_MIN_Z + z_offset

        p.resetBasePositionAndOrientation(
            laser_id,
            [base_x, base_y, new_z],
            [0, 0, 0, 1]
        )


    # --------------------------------------------------
    # Move base forward
    # --------------------------------------------------
    p.setJointMotorControl2(
        robot,
        joint_map["joint_mobile_x"],
        p.VELOCITY_CONTROL,
        targetVelocity=1.0,
        force=100000
    )

    # --------------------------------------------------
    # Arm waving motion
    # --------------------------------------------------
    angle1 = 0.8 * math.sin(step * 0.02)
    angle2 = 0.6 * math.sin(step * 0.02 + 1.0)

    p.setJointMotorControl2(
        robot,
        arm_joints[0],
        p.POSITION_CONTROL,
        targetPosition=angle1,
        force=500
    )

    p.setJointMotorControl2(
        robot,
        arm_joints[1],
        p.POSITION_CONTROL,
        targetPosition=angle2,
        force=500
    )

    # --------------------------------------------------
    # Step simulation
    # --------------------------------------------------
    p.stepSimulation()
    time.sleep(1 / 240)


# ======================================================================
# STOP MOTION AFTER SIM
# ======================================================================

# Stop base
p.setJointMotorControl2(
    robot,
    joint_map["joint_mobile_x"],
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=100
)

# Reset arm
for jid in arm_joints:
    p.setJointMotorControl2(
        robot,
        jid,
        p.POSITION_CONTROL,
        targetPosition=0,
        force=500
    )

time.sleep(2)
