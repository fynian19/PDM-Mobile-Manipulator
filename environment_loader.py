import pybullet as p
import pybullet_data
import math


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _create_box(pos, half_extents, color, orientation=None):
    col = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents
    )
    vis = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )

    if orientation is None:
        orientation = [0, 0, 0, 1]

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=orientation
    )


def _create_cylinder(pos, radius, height, color, orientation=None):
    col = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )
    vis = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )

    if orientation is None:
        orientation = [0, 0, 0, 1]

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=orientation
    )


def _create_sphere(pos, radius, color):
    col = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius
    )
    vis = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color
    )

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos
    )


def _parse_color(parts, start_idx):
    r = float(parts[start_idx])
    g = float(parts[start_idx + 1])
    b = float(parts[start_idx + 2])

    if len(parts) > start_idx + 3:
        a = float(parts[start_idx + 3])
        return [r, g, b, a], start_idx + 4

    return [r, g, b, 1.0], start_idx + 3


def _parse_orientation(parts, start_idx):
    if len(parts) >= start_idx + 3:
        roll = float(parts[start_idx])
        pitch = float(parts[start_idx + 1])
        yaw = float(parts[start_idx + 2])
        return p.getQuaternionFromEuler([roll, pitch, yaw])
    return None


# ======================================================================
# ENVIRONMENT LOADER
# ======================================================================

def load_environment_from_txt(path):
    obstacle_ids = []
    laser_ids = []
    laser_base_positions = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            shape = parts[0].upper()

            # ------------------- BOX -------------------
            if shape == "BOX":
                px, py, pz = map(float, parts[1:4])
                sx, sy, sz = map(float, parts[4:7])

                color, idx = _parse_color(parts, 7)
                orientation = _parse_orientation(parts, idx)

                obstacle_ids.append(
                    _create_box(
                        [px, py, pz],
                        [sx, sy, sz],
                        color,
                        orientation
                    )
                )

            # ------------------- LASER BOX -------------------
            elif shape == "LASER_BOX":
                px, py, pz = map(float, parts[1:4])
                sx, sy, sz = map(float, parts[4:7])

                color, idx = _parse_color(parts, 7)
                orientation = _parse_orientation(parts, idx)

                laser_id = _create_box(
                    [px, py, pz],
                    [sx, sy, sz],
                    color,
                    orientation
                )

                laser_ids.append(laser_id)
                laser_base_positions.append([px, py, pz])

            # ------------------- CYLINDER -------------------
            elif shape == "CYL":
                px, py, pz = map(float, parts[1:4])
                radius = float(parts[4])
                height = float(parts[5])

                color, idx = _parse_color(parts, 6)
                orientation = _parse_orientation(parts, idx)

                obstacle_ids.append(
                    _create_cylinder(
                        [px, py, pz],
                        radius,
                        height,
                        color,
                        orientation
                    )
                )

            # ------------------- SPHERE -------------------
            elif shape == "SPH":
                px, py, pz = map(float, parts[1:4])
                radius = float(parts[4])
                color, _ = _parse_color(parts, 5)

                obstacle_ids.append(
                    _create_sphere([px, py, pz], radius, color)
                )

            else:
                print(f"[WARN] Unknown shape type: {shape}")

    return obstacle_ids, laser_ids, laser_base_positions
