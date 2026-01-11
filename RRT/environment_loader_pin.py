import coal
import pinocchio
from pyroboplan.core.utils import set_collisions
import numpy as np
from scipy.spatial.transform import Rotation as R  # for orientation


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _create_box(idx, pos, half_extents, color, orientation, vm, cm):

    obstacle = pinocchio.GeometryObject(
    "obstacle_"+str(idx),
    0,
    pinocchio.SE3(orientation, np.array(pos)),
    coal.Box(half_extents[0]*2, half_extents[1]*2, half_extents[2]*2)
    )
    obstacle.meshColor = np.array(color)
    vm.addGeometryObject(obstacle)
    cm.addGeometryObject(obstacle)
    
    if color == [0.0, 1.0, 0.0, 1.0]:  # green color of the goal object
        obstacle.name = "goal"
    elif color == [0.7, 0.10, 0.95, 1.0]: # purple alternative color
        obstacle.name = "goal"
    		
    return obstacle


def _create_cylinder(idx, pos, radius, height, color, orientation, vm, cm):

    obstacle = pinocchio.GeometryObject(
    "obstacle_"+str(idx),
    0,
    pinocchio.SE3(orientation, np.array(pos)),
    coal.Cylinder(radius, height)
    )
    obstacle.meshColor = np.array(color)
    vm.addGeometryObject(obstacle)
    cm.addGeometryObject(obstacle)
    
    return obstacle


def _create_sphere(idx, pos, radius, color, orientation, vm, cm):

    obstacle = pinocchio.GeometryObject(
    "obstacle_"+str(idx),
    0,
    pinocchio.SE3(orientation, np.array(pos)),
    coal.Sphere(radius)
    )
    obstacle.meshColor = np.array(color)
    vm.addGeometryObject(obstacle)
    cm.addGeometryObject(obstacle)
    
    return obstacle


def _parse_color(parts, start_idx):
    r = float(parts[start_idx])
    g = float(parts[start_idx + 1])
    b = float(parts[start_idx + 2])

    if len(parts) > start_idx + 3:
        a = float(parts[start_idx + 3])
        return [r, g, b, a], start_idx + 4

    return [r, g, b, 1.0], start_idx + 3


def _parse_orientation(parts, start_idx):
    """
    Optional roll pitch yaw (radians).
    If not present, return identity quaternion.
    """
    if len(parts) >= start_idx + 3:
        roll = float(parts[start_idx])
        pitch = float(parts[start_idx + 1])
        yaw = float(parts[start_idx + 2])
        r = R.from_euler('ZYX', [yaw, pitch, roll])
        return np.array(r.as_matrix())
    return np.eye(3)


# ======================================================================
# ENVIRONMENT LOADER
# ======================================================================

def load_environment_from_txt(path, vm, cm):
    """
    Reads an obstacle description file and creates all objects in PyBullet.

    Supported formats:
    BOX px py pz sx sy sz r g b
    CYL px py pz radius height r g b
    SPH px py pz radius r g b

    Returns:
        list of created body unique names
    """

    obstacle_ids = []
    idx = 1

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            shape = parts[0].upper()

            # ------------------- BOX -------------------
            if shape == "BOX":
                #_, px, py, pz, sx, sy, sz, r, g, b = parts
                #pos = [float(px), float(py), float(pz)]
                #sx, sy, sz = float(sx), float(sy), float(sz)
                #color = [float(r), float(g), float(b), 1.0]
                
		# BOX px py pz sx sy sz r g b [a] [roll pitch yaw]
                px, py, pz = map(float, parts[1:4])
                sx, sy, sz = map(float, parts[4:7])

                color, idx_parts = _parse_color(parts, 7)
                orientation = _parse_orientation(parts, idx_parts)
                
                obstacle_ids.append(_create_box(idx, [px, py, pz], [sx, sy, sz], color, orientation, vm, cm).name)
                
                idx += 1


            # ------------------- CYLINDER -------------------
            elif shape == "CYL":
                #_, px, py, pz, radius, height, r, g, b = parts
                #pos = [float(px), float(py), float(pz)]
                #radius = float(radius)
                #height = float(height)
                #color = [float(r), float(g), float(b), 1.0]
                
                px, py, pz = map(float, parts[1:4])
                radius = float(parts[4])
                height = float(parts[5])

                color, idx_parts = _parse_color(parts, 6)
                orientation = _parse_orientation(parts, idx_parts)
                
                obstacle_ids.append(_create_cylinder(idx, [px, py, pz], radius, height, color, orientation, vm, cm).name)
                
                idx += 1

            # ------------------- SPHERE -------------------
            elif shape == "SPH":
                #_, px, py, pz, radius, r, g, b = parts
                #pos = [float(px), float(py), float(pz)]
                #radius = float(radius)
                #color = [float(r), float(g), float(b), 1.0]
                px, py, pz = map(float, parts[1:4])
                radius = float(parts[4])
                color, _ = _parse_color(parts, 5)
                
                obstacle_ids.append(_create_sphere(idx, [px, py, pz], radius, color, vm, cm).name)
                
                idx += 1

            else:
                print(f"[WARN] Unknown shape type: {shape}")

    return obstacle_ids
