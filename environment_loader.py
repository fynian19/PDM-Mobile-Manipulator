import pybullet as p
import pybullet_data


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _create_box(pos, half_extents, color):
    col = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents
    )
    vis = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos
    )


def _create_cylinder(pos, radius, height, color):
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
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos
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


# ======================================================================
# ENVIRONMENT LOADER
# ======================================================================

def load_environment_from_txt(path):
    """
    Reads an obstacle description file and creates all objects in PyBullet.

    Supported formats:
    BOX px py pz sx sy sz r g b
    CYL px py pz radius height r g b
    SPH px py pz radius r g b

    Returns:
        list of created body unique IDs
    """

    obstacle_ids = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            shape = parts[0].upper()

            # ------------------- BOX -------------------
            if shape == "BOX":
                _, px, py, pz, sx, sy, sz, r, g, b = parts
                pos = [float(px), float(py), float(pz)]
                sx, sy, sz = float(sx), float(sy), float(sz)
                color = [float(r), float(g), float(b), 1.0]
                obstacle_ids.append(_create_box(pos, [sx, sy, sz], color))

            # ------------------- CYLINDER -------------------
            elif shape == "CYL":
                _, px, py, pz, radius, height, r, g, b = parts
                pos = [float(px), float(py), float(pz)]
                radius = float(radius)
                height = float(height)
                color = [float(r), float(g), float(b), 1.0]
                obstacle_ids.append(_create_cylinder(pos, radius, height, color))

            # ------------------- SPHERE -------------------
            elif shape == "SPH":
                _, px, py, pz, radius, r, g, b = parts
                pos = [float(px), float(py), float(pz)]
                radius = float(radius)
                color = [float(r), float(g), float(b), 1.0]
                obstacle_ids.append(_create_sphere(pos, radius, color))

            else:
                print(f"[WARN] Unknown shape type: {shape}")

    return obstacle_ids
