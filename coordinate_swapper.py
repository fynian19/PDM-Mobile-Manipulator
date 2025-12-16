import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

# ======================================================================
# ROOM
# ======================================================================

HX = 10.0
HY = 10.0
Z_MIN = 0.0
Z_MAX = 2.2

NUM_LASERS = 30
LASER_RADIUS = 0.03
RGBA = (1, 0, 0, 1)

Z_THRESHOLD = 0.5

# ======================================================================
# EXCLUSION ZONES (expanded safety margins)
# ======================================================================

EXCLUSION_MARGIN = 0.4  # how far lasers must stay away

# Window box: BOX -9.9 -8.5 1.5  0.05 0.8 0.8
WINDOW_CENTER = np.array([-9.9, -8.5, 1.5])
WINDOW_HALF = np.array([0.05, 0.8, 0.8]) + EXCLUSION_MARGIN

# Goal pedestal: BOX 0 8.0 0.25  0.30 0.30 0.50
GOAL_CENTER = np.array([0.0, 8.0, 0.25])
GOAL_HALF = np.array([0.30, 0.30, 0.50]) + EXCLUSION_MARGIN

def point_in_box(p, center, half_extents):
    return np.all(np.abs(p - center) <= half_extents)

# ======================================================================
# HELPERS
# ======================================================================

def random_point_on_plane():
    plane = random.choice(["x-", "x+", "y-", "y+", "z0", "z1"])

    if plane == "x-":
        return np.array([-HX, random.uniform(-HY, HY), random.uniform(Z_MIN, Z_MAX)])
    if plane == "x+":
        return np.array([ HX, random.uniform(-HY, HY), random.uniform(Z_MIN, Z_MAX)])
    if plane == "y-":
        return np.array([random.uniform(-HX, HX), -HY, random.uniform(Z_MIN, Z_MAX)])
    if plane == "y+":
        return np.array([random.uniform(-HX, HX),  HY, random.uniform(Z_MIN, Z_MAX)])
    if plane == "z0":
        return np.array([random.uniform(-HX, HX), random.uniform(-HY, HY), Z_MIN])
    if plane == "z1":
        return np.array([random.uniform(-HX, HX), random.uniform(-HY, HY), Z_MAX])

PLANES = ["x-", "x+", "y-", "y+", "z0", "z1"]

def random_point_on_specific_plane(plane):
    if plane == "x-":
        return np.array([-HX, random.uniform(-HY, HY), random.uniform(Z_MIN, Z_MAX)])
    if plane == "x+":
        return np.array([ HX, random.uniform(-HY, HY), random.uniform(Z_MIN, Z_MAX)])
    if plane == "y-":
        return np.array([random.uniform(-HX, HX), -HY, random.uniform(Z_MIN, Z_MAX)])
    if plane == "y+":
        return np.array([random.uniform(-HX, HX),  HY, random.uniform(Z_MIN, Z_MAX)])
    if plane == "z0":
        return np.array([random.uniform(-HX, HX), random.uniform(-HY, HY), Z_MIN])
    if plane == "z1":
        return np.array([random.uniform(-HX, HX), random.uniform(-HY, HY), Z_MAX])


def quat_from_two_vectors(v0, v1):
    """Quaternion rotating v0 -> v1"""
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    dot = np.dot(v0, v1)

    if dot > 0.999999:
        return np.array([0, 0, 0, 1])

    if dot < -0.999999:
        # 180° rotation around any orthogonal axis
        axis = np.cross(v0, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v0, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0])

    axis = np.cross(v0, v1)
    s = math.sqrt((1 + dot) * 2)
    invs = 1 / s

    return np.array([
        axis[0] * invs,
        axis[1] * invs,
        axis[2] * invs,
        s * 0.5
    ])


def endpoints_to_cyl(p0, p1):
    center = (p0 + p1) / 2.0
    d = p1 - p0
    length = np.linalg.norm(d)
    d /= length

    # Rotate +Z into direction d
    q = quat_from_two_vectors(np.array([0, 0, 1]), d)

    # Convert quaternion to roll-pitch-yaw
    roll, pitch, yaw = p.getEulerFromQuaternion(q)

    return center, LASER_RADIUS, length, RGBA, (roll, pitch, yaw)

# ======================================================================
# GENERATE
# ======================================================================

lasers = []
segments = []

for _ in range(NUM_LASERS):
    while True:
        plane0, plane1 = random.sample(PLANES, 2)

        p0 = random_point_on_specific_plane(plane0)
        p1 = random_point_on_specific_plane(plane1)

        # Reject if BOTH points are below height threshold
        if p0[2] < Z_THRESHOLD and p1[2] < Z_THRESHOLD:
            continue

        # Reject if near window
        if (
            point_in_box(p0, WINDOW_CENTER, WINDOW_HALF) or
            point_in_box(p1, WINDOW_CENTER, WINDOW_HALF)
        ):
            continue

        # Reject if near goal
        if (
            point_in_box(p0, GOAL_CENTER, GOAL_HALF) or
            point_in_box(p1, GOAL_CENTER, GOAL_HALF)
        ):
            continue

        break

    lasers.append(endpoints_to_cyl(p0, p1))
    segments.append((p0, p1))




# ======================================================================
# OUTPUT
# ======================================================================

print("# ---- CYL OUTPUT (PyBullet correct) ----")
for pos, r, h, rgba, rot in lasers:
    print(
        f"CYL {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} "
        f"{r:.3f} {h:.3f} "
        f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]} "
        f"{rot[0]:.6f} {rot[1]:.6f} {rot[2]:.6f}"
    )

# ======================================================================
# VISUAL DEBUG
# ======================================================================

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

xs = [-HX, HX]
ys = [-HY, HY]
zs = [Z_MIN, Z_MAX]

for x in xs:
    for y in ys:
        ax.plot([x, x], [y, y], zs, color="gray")
for x in xs:
    for z in zs:
        ax.plot([x, x], ys, [z, z], color="gray")
for y in ys:
    for z in zs:
        ax.plot(xs, [y, y], [z, z], color="gray")

for p0, p1 in segments:
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="red")

ax.set_xlim(-HX, HX)
ax.set_ylim(-HY, HY)
ax.set_zlim(Z_MIN, Z_MAX)
ax.set_title("Lasers (Correct Orientation)")

plt.tight_layout()
plt.show()
