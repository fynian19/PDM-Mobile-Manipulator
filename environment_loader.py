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


def _parse_color(parts, start_idx):
    """
    Parses r g b [a] starting at start_idx.
    If alpha is missing, defaults to 1.0.
    """
    r = float(parts[start_idx])
    g = float(parts[start_idx + 1])
    b = float(parts[start_idx + 2])

    if len(parts) > start_idx + 3:
        a = float(parts[start_idx + 3])
    else:
        a = 1.0

    return [r, g, b, a]


# ======================================================================
# ENVIRONMENT LOADER
# ======================================================================

def load_environment_from_txt(path):
    """
    Reads an obstacle description file and creates all objects in PyBullet.

    Supported formats:
    BOX px py pz sx sy sz r g b [a]
    CYL px py pz radius height r g b [a]
    SPH px py pz radius r g b [a]

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
                # BOX px py pz sx sy sz r g b [a]
                px, py, pz = map(float, parts[1:4])
                sx, sy, sz = map(float, parts[4:7])
                color = _parse_color(parts, 7)

                obstacle_ids.append(
                    _create_box([px, py, pz], [sx, sy, sz], color)
                )

            # ------------------- CYLINDER -------------------
            elif shape == "CYL":
                # CYL px py pz radius height r g b [a]
                px, py, pz = map(float, parts[1:4])
                radius = float(parts[4])
                height = float(parts[5])
                color = _parse_color(parts, 6)

                obstacle_ids.append(
                    _create_cylinder([px, py, pz], radius, height, color)
                )

            # ------------------- SPHERE -------------------
            elif shape == "SPH":
                # SPH px py pz radius r g b [a]
                px, py, pz = map(float, parts[1:4])
                radius = float(parts[4])
                color = _parse_color(parts, 5)

                obstacle_ids.append(
                    _create_sphere([px, py, pz], radius, color)
                )

            else:
                print(f"[WARN] Unknown shape type: {shape}")

    return obstacle_ids
