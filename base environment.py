import pybullet as p
import pybullet_data
import time
import math

# Connect to PyBullet
p.connect(p.GUI)

# Hide PyBullet debug UI clutter
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

# Load environment
p.loadURDF("plane.urdf")
robot = p.loadURDF("URDF/mobileManipulator.urdf")

# -------- ADD OBSTACLES HERE --------

def create_box_obstacle(position, half_extents, color):
    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )
    body = p.createMultiBody(
        baseMass=0,  # static obstacle
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position
    )
    return body

# Wall in front
create_box_obstacle(
    position=[3, 0, 0.5],
    half_extents=[0.1, 2.0, 0.5],
    color=[1, 0, 0, 1]
)

# Side block
create_box_obstacle(
    position=[1.5, 1.5, 0.5],
    half_extents=[0.5, 0.5, 0.5],
    color=[0, 0, 1, 1]
)

# Pillar
create_box_obstacle(
    position=[2.0, -1.0, 1.0],
    half_extents=[0.3, 0.3, 1.0],
    color=[0, 1, 0, 1]
)

# -------- Build joint map --------
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

# Arm joint IDs (renamed)
arm_joints = [
    joint_map["joint_base_to_upper_arm"],
    joint_map["joint_upper_to_lower_arm"]
]

# -------- Main simulation loop --------
for step in range(800):
    # Move base forward
    p.setJointMotorControl2(
        robot,
        joint_map["joint_mobile_x"],
        p.VELOCITY_CONTROL,
        targetVelocity=1.0,
        force=100000
    )

    # Arm waving motion
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

    # Step simulation
    p.stepSimulation()
    time.sleep(1 / 240)

# -------- Stop motion --------
p.setJointMotorControl2(
    robot,
    joint_map["joint_mobile_x"],
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=100
)

for jid in arm_joints:
    p.setJointMotorControl2(
        robot,
        jid,
        p.POSITION_CONTROL,
        targetPosition=0,
        force=500
    )

time.sleep(2)
