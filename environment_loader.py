import pybullet as p

def _create_box(pos, half_extents, color):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(0, col, vis, basePosition=pos)

def _create_cylinder(pos, radius, height, color):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return p.createMultiBody(0, col, vis, basePosition=pos)

def load_environment_from_txt(path):
    """
    Parses obstacles.txt and returns:
    1. ids: List of PyBullet Body IDs
    2. data: List of dicts with math data for MPC {'type', 'pos', 'size'/'radius'}
    """
    obstacle_ids = []
    obstacle_data = [] 

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue

            parts = line.split()
            shape = parts[0].upper()

            if shape == "BOX":
                # Format: BOX px py pz sx sy sz r g b
                # Note: sx, sy, sz are HALF-extents in PyBullet logic
                _, px, py, pz, sx, sy, sz, r, g, b = parts
                pos = [float(px), float(py), float(pz)]
                half_extents = [float(sx), float(sy), float(sz)]
                color = [float(r), float(g), float(b), 1.0]
                
                obstacle_ids.append(_create_box(pos, half_extents, color))
                
                obstacle_data.append({
                    'type': 'BOX',
                    'pos': pos[:2],      # Keep only x,y for MPC
                    'size': half_extents[:2] # Keep only sx, sy
                })

            elif shape == "CYL":
                # Format: CYL px py pz radius height r g b
                _, px, py, pz, rad, h, r, g, b = parts
                pos = [float(px), float(py), float(pz)]
                radius = float(rad)
                height = float(h)
                color = [float(r), float(g), float(b), 1.0]
                
                obstacle_ids.append(_create_cylinder(pos, radius, height, color))
                
                obstacle_data.append({
                    'type': 'CYL',
                    'pos': pos[:2],
                    'radius': radius
                })

    return obstacle_ids, obstacle_data