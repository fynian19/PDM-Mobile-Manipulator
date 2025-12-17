import pybullet as p
import numpy as np

# ==========================================
# 1. VISUAL/COLLISION HELPERS
# ==========================================

def _create_box(pos, half_extents, color, orientation=None):
    if orientation is None: orientation = [0, 0, 0, 1]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(0, col, vis, basePosition=pos, baseOrientation=orientation)

def _create_cylinder(pos, radius, height, color, orientation=None):
    if orientation is None: orientation = [0, 0, 0, 1]
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return p.createMultiBody(0, col, vis, basePosition=pos, baseOrientation=orientation)

def _parse_color(parts, start_idx):
    """
    Robustly parses RGB or RGBA starting at start_idx.
    Returns: (color_list, next_index)
    """
    r = float(parts[start_idx])
    g = float(parts[start_idx+1])
    b = float(parts[start_idx+2])
    
    # Check if Alpha exists (if array is long enough)
    if len(parts) > start_idx + 3:
        # It might be alpha, check if it looks like a float 0-1
        # But wait, orientation (roll) could also be a float. 
        # Heuristic: If there are 3 more floats after this, this is alpha. 
        # If there are exactly 3 floats left including this one, it's orientation.
        remaining = len(parts) - (start_idx + 3)
        if remaining == 0 or remaining >= 3:
             # Likely Alpha exists
             a = float(parts[start_idx+3])
             return [r, g, b, a], start_idx + 4
             
    # Default Alpha = 1.0
    return [r, g, b, 1.0], start_idx + 3

def _parse_orientation(parts, start_idx):
    """
    Parses optional [Roll Pitch Yaw]. Returns Quaternion or Identity.
    """
    if len(parts) >= start_idx + 3:
        roll = float(parts[start_idx])
        pitch = float(parts[start_idx+1])
        yaw = float(parts[start_idx+2])
        return p.getQuaternionFromEuler([roll, pitch, yaw])
    return [0, 0, 0, 1] # Identity

# ==========================================
# 2. MATH HELPERS FOR MPC
# ==========================================

def get_cylinder_endpoints(pos, height, orientation_quat):
    """
    Calculates 3D start/end points of a cylinder for the MPC.
    """
    # Default cylinder axis is Z [0,0,1]
    axis_vector = np.array([0, 0, 1])
    
    # Rotate axis by quaternion
    rot_matrix = np.array(p.getMatrixFromQuaternion(orientation_quat)).reshape(3, 3)
    rotated_axis = rot_matrix @ axis_vector
    
    half_h = height / 2.0
    p0 = np.array(pos) - rotated_axis * half_h
    p1 = np.array(pos) + rotated_axis * half_h
    return p0, p1

# ==========================================
# 3. MAIN LOADER FUNCTION
# ==========================================

def load_environment_from_txt(path):
    """
    Parses complex environment files.
    Returns:
       obstacle_ids: List of PyBullet IDs
       obstacle_data: List of dicts for MPC {'type', 'p0', 'p1', 'radius'}
    """
    obstacle_ids = []
    obstacle_data = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue

            parts = line.split()
            shape = parts[0].upper()

            # ------------------- BOX -------------------
            if shape == "BOX":
                # BOX px py pz sx sy sz r g b [a] [roll pitch yaw]
                px, py, pz = map(float, parts[1:4])
                sx, sy, sz = map(float, parts[4:7])
                
                # Use robust parser for variable length
                color, next_idx = _parse_color(parts, 7)
                ori = _parse_orientation(parts, next_idx)

                pos = [px, py, pz]
                half_extents = [sx, sy, sz]
                
                obstacle_ids.append(_create_box(pos, half_extents, color, ori))
                
                # Math Data for MPC
                # Note: We treat boxes as axis-aligned for simplicity in MPC
                # (Complex rotated boxes require convex hull constraints)
                obstacle_data.append({
                    'type': 'BOX',
                    'pos': pos[:2],      
                    'size': half_extents[:2] 
                })

            # ------------------- CYLINDER -------------------
            elif shape == "CYL":
                # CYL px py pz radius height r g b [a] [roll pitch yaw]
                px, py, pz = map(float, parts[1:4])
                radius = float(parts[4])
                height = float(parts[5])

                color, next_idx = _parse_color(parts, 6)
                ori = _parse_orientation(parts, next_idx)
                
                pos = [px, py, pz]
                
                obstacle_ids.append(_create_cylinder(pos, radius, height, color, ori))
                
                # Math Data for MPC (Endpoints)
                p0, p1 = get_cylinder_endpoints(pos, height, ori)
                
                obstacle_data.append({
                    'type': 'CYL',
                    'radius': radius,
                    'p0': p0, 
                    'p1': p1
                })

    return obstacle_ids, obstacle_data