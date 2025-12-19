import pybullet as p
import numpy as np

def _create_box(pos, half_extents, color, orientation=None, ghost=False):
    if orientation is None: orientation = [0, 0, 0, 1]
    
    if ghost:
        # GHOST: No collision shape (Collision ID = -1)
        col = -1 
        # Optional: Make it semi-transparent
        color = [color[0], color[1], color[2], 0.5]
    else:
        # PHYSICAL: Normal collision shape
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(0, col, vis, basePosition=pos, baseOrientation=orientation)

def _create_cylinder(pos, radius, height, color, orientation=None):
    if orientation is None: orientation = [0, 0, 0, 1]
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return p.createMultiBody(0, col, vis, basePosition=pos, baseOrientation=orientation)

def _parse_color(parts, start_idx):
    r = float(parts[start_idx])
    g = float(parts[start_idx+1])
    b = float(parts[start_idx+2])
    if len(parts) > start_idx + 3:
        remaining = len(parts) - (start_idx + 3)
        if remaining == 0 or remaining >= 3:
             a = float(parts[start_idx+3])
             return [r, g, b, a], start_idx + 4
    return [r, g, b, 1.0], start_idx + 3

def _parse_orientation(parts, start_idx):
    if len(parts) >= start_idx + 3:
        roll = float(parts[start_idx])
        pitch = float(parts[start_idx+1])
        yaw = float(parts[start_idx+2])
        return p.getQuaternionFromEuler([roll, pitch, yaw])
    return [0, 0, 0, 1]

def get_cylinder_endpoints(pos, height, orientation_quat):
    axis_vector = np.array([0, 0, 1])
    rot_matrix = np.array(p.getMatrixFromQuaternion(orientation_quat)).reshape(3, 3)
    rotated_axis = rot_matrix @ axis_vector
    half_h = height / 2.0
    p0 = np.array(pos) - rotated_axis * half_h
    p1 = np.array(pos) + rotated_axis * half_h
    return p0, p1

def load_environment_from_txt(path):
    obstacle_ids = []
    obstacle_data = []

    # === TARGET DEFINITION ===
    # This matches the Z-height of the top box (0.875)
    # The Pedestal is at Z=0.25, so it won't trigger this.
    GOAL_CUBE_POS = np.array([0.0, 8.0, 0.875])

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue

            parts = line.split()
            shape = parts[0].upper()

            if shape == "BOX":
                px, py, pz = map(float, parts[1:4])
                sx, sy, sz = map(float, parts[4:7])
                color, next_idx = _parse_color(parts, 7)
                ori = _parse_orientation(parts, next_idx)

                pos = [px, py, pz]
                half_extents = [sx, sy, sz]
                
                # --- 3D DISTANCE CHECK ---
                # This ensures we only pick the top box, not the pedestal below it.
                dist = np.linalg.norm(np.array(pos) - GOAL_CUBE_POS)
                is_target_ghost = (dist < 0.1) 
                
                # Create Body (Physics handled inside _create_box based on flag)
                body_id = _create_box(pos, half_extents, color, ori, ghost=is_target_ghost)
                obstacle_ids.append(body_id)
                
                obstacle_data.append({
                    'type': 'BOX',
                    'pos': pos,
                    'size': half_extents,
                    'is_ghost': is_target_ghost  # Pass this flag to the Main Script
                })

            elif shape == "CYL":
                px, py, pz = map(float, parts[1:4])
                radius = float(parts[4])
                height = float(parts[5])
                color, next_idx = _parse_color(parts, 6)
                ori = _parse_orientation(parts, next_idx)
                
                pos = [px, py, pz]
                body_id = _create_cylinder(pos, radius, height, color, ori)
                obstacle_ids.append(body_id)
                
                p0, p1 = get_cylinder_endpoints(pos, height, ori)
                
                obstacle_data.append({
                    'type': 'CYL',
                    'pos': pos,
                    'radius': radius,
                    'p0': p0, 'p1': p1,
                    'is_ghost': False # Cylinders are never ghosts in this setup
                })

    return obstacle_ids, obstacle_data