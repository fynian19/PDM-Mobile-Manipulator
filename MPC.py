import numpy as np
import pinocchio as pin
import cvxpy as cp
import scipy.linalg
from eq_motion_derivation import discretize_dynamics, get_linear_dynamics

# ==========================================
# 1. GEOMETRY HELPERS (Exact Distance)
# ==========================================

def get_transform(pos, quat):
    """
    Creates 4x4 Homogeneous Matrix.
    Robustly handles 2D (x,y) or 3D (x,y,z) positions.
    """
    T = np.eye(4)
    T[:3, :3] = pin.Quaternion(np.array(quat)).matrix()
    
    # FIX: Ensure pos is 3D
    if len(pos) == 2:
        T[:2, 3] = pos
        T[2, 3] = 0.0 # Default Z=0
    else:
        T[:3, 3] = pos
        
    return T

def dist_point_to_box(point, box_pos, box_quat, box_size):
    """
    Computes exact distance and normal from a 3D point to an Oriented Box.
    """
    # Transform point to Box Frame
    T_world_box = get_transform(box_pos, box_quat)
    # Fast inverse for homogenous matrix (Rotation^T)
    R_T = T_world_box[:3, :3].T
    p_local = R_T @ (point - T_world_box[:3, 3])
    
    # FIX: Ensure box_size is 3D (pad with arbitrary height if 2D)
    if len(box_size) == 2:
        size_3d = np.array([box_size[0], box_size[1], 1.0])
    else:
        size_3d = np.array(box_size)

    # Clamp to find closest point on surface
    closest_local = np.clip(p_local, -size_3d, size_3d)
    
    # Distance and Normal in Local Frame
    diff_local = p_local - closest_local
    dist = np.linalg.norm(diff_local)
    
    if dist < 1e-6:
        # Inside box: push out along Z axis (fallback) or X
        normal_local = np.array([1.0, 0, 0])
        dist = -1e-4
    else:
        normal_local = diff_local / dist

    # Rotate Normal to World Frame
    normal_world = T_world_box[:3, :3] @ normal_local
    return dist, normal_world

def dist_point_to_cylinder(point, cyl_pos, cyl_quat, radius, height):
    # Transform to Cylinder Frame
    T_world_box = get_transform(cyl_pos, cyl_quat)
    R_T = T_world_box[:3, :3].T
    p_local = R_T @ (point - T_world_box[:3, 3])
    
    # Project to Axis (Z)
    z_clamped = np.clip(p_local[2], -height/2, height/2)
    axis_point = np.array([0, 0, z_clamped])
    
    # Radial distance
    vec_radial = p_local - axis_point
    dist_radial = np.linalg.norm(vec_radial)
    
    if dist_radial < 1e-6:
        normal_local = np.array([1.0, 0, 0])
    else:
        normal_local = vec_radial / dist_radial
        
    dist = dist_radial - radius
    normal_world = T_world_box[:3, :3] @ normal_local
    return dist, normal_world

def get_clamped_reference(x_current, x_global, max_lookahead=3.0):
    curr_pos = x_current[:2]
    glob_pos = x_global[:2]
    diff = glob_pos - curr_pos
    dist = np.linalg.norm(diff)
    
    if dist > max_lookahead:
        direction = diff / dist
        local_pos = curr_pos + direction * max_lookahead
        x_ref_local = x_global.copy()
        x_ref_local[0] = local_pos[0]
        x_ref_local[1] = local_pos[1]
        return x_ref_local
    else:
        return x_global

# ==========================================
# 2. MPC CONTROLLER CLASS
# ==========================================

class MPCController:
    def __init__(self, urdf_path, x_ref, dt, N=10, obstacle_list=None, bounds=None):
        self.dt = dt
        self.N = N 
        self.x_ref_val = x_ref
        self.obstacle_list = obstacle_list if obstacle_list else []
        self.bounds = bounds if bounds else {'x_min': -10, 'x_max': 10, 'y_min': -10, 'y_max': 10}
        
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData() 
        self.nx = self.model.nv * 2
        self.nu = self.model.nv

        self.Q_diag = np.array([100, 100, 50, 50, 50] + [10, 10, 10, 5, 5]) 
        self.R_diag = np.array([0.5] * self.nu) 
        
        # --- FIX: ROBUST FRAME ITERATION ---
        self.collision_frames = []
        if self.model.existFrame("base_link"):
            self.collision_frames.append(self.model.getFrameId("base_link"))
        
        for i, f in enumerate(self.model.frames):
            if "link" in f.name and "base" not in f.name:
                self.collision_frames.append(i)
        
        self.MAX_CONSTRAINTS = 10 
        
        self._compute_terminal_cost()
        self._setup_mpc_problem()

    def _compute_terminal_cost(self):
        q_goal = self.x_ref_val[:self.model.nq]
        v_goal = np.zeros(self.model.nv)
        Ac, Bc, _ = get_linear_dynamics(q_goal, v_goal, np.zeros(self.nu), self.model, self.data)
        Ad = np.eye(self.nx) + Ac * self.dt
        Bd = Bc * self.dt
        self.P_val = scipy.linalg.solve_discrete_are(Ad, Bd, np.diag(self.Q_diag), np.diag(self.R_diag))

    def _get_tangent_plane_constraints(self, q_current):
        """
        Generates Linear Constraints: C @ q_next >= d
        """
        # Update Kinematics
        pin.forwardKinematics(self.model, self.data, q_current)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_current)

        candidates = []

        # 1. Loop over Robot Joints
        for frame_idx in self.collision_frames:
            p_robot = self.data.oMf[frame_idx].translation
            # Get Jacobian (Translation part 3xN)
            J_6d = pin.getFrameJacobian(self.model, self.data, frame_idx, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_lin = J_6d[:3, :] 

            # 2. Loop over Obstacles
            for obs in self.obstacle_list:
                # --- EXACT GEOMETRY CHECK ---
                if obs['type'] == 'BOX':
                    # FIX: Handle 2D or 3D pos gracefully
                    raw_pos = np.array(obs.get('pos', [0,0,0]))
                    if len(raw_pos) == 2: pos = np.array([raw_pos[0], raw_pos[1], 0.0])
                    else: pos = raw_pos
                    
                    quat = obs.get('quat', [0,0,0,1])
                    size = np.array(obs.get('size', [0.1,0.1,0.1]))
                    
                    dist, normal = dist_point_to_box(p_robot, pos, quat, size)
                    
                elif obs['type'] == 'CYL':
                    if 'pos' in obs: 
                        raw_pos = np.array(obs['pos'])
                        if len(raw_pos) == 2: pos = np.array([raw_pos[0], raw_pos[1], 0.0])
                        else: pos = raw_pos
                    else: 
                        # Center from endpoints
                        p0 = np.array(obs.get('p0', [0,0,0]))
                        p1 = np.array(obs.get('p1', [0,0,1]))
                        # Pad if p0/p1 are 2D
                        if len(p0) == 2: p0 = np.append(p0, 0.0)
                        if len(p1) == 2: p1 = np.append(p1, 1.0)
                        pos = (p0+p1)/2.0
                    
                    quat = obs.get('quat', [0,0,0,1])
                    radius = obs.get('radius', 0.5)
                    height = obs.get('height', 1.0)
                    dist, normal = dist_point_to_cylinder(p_robot, pos, quat, radius, height)
                
                # --- TANGENT PLANE CONSTRAINT ---
                d_safe = 0.2 
                
                # Gradient C = n^T * J
                C_row = normal @ J_lin 
                
                dist_clamped = max(dist, -0.05) 
                d_val = d_safe - dist_clamped + np.dot(C_row, q_current)

                candidates.append((dist, C_row, d_val))

        # 3. Filter and Sort
        candidates.sort(key=lambda x: x[0]) 
        
        C_out = np.zeros((self.MAX_CONSTRAINTS, self.model.nq))
        d_out = np.full(self.MAX_CONSTRAINTS, -1000.0) 
        
        for i in range(min(len(candidates), self.MAX_CONSTRAINTS)):
            C_out[i, :] = candidates[i][1]
            d_out[i]    = candidates[i][2]
            
        return C_out, d_out

    def _setup_mpc_problem(self):
        # Params
        self.p_Ad = cp.Parameter((self.nx, self.nx), name="Ad")
        self.p_Bd = cp.Parameter((self.nx, self.nu), name="Bd")
        self.p_dd = cp.Parameter((self.nx,), name="dd")
        self.p_x0 = cp.Parameter((self.nx,), name="x0")
        self.p_xref = cp.Parameter((self.nx,), name="xref")
        
        # Tangent Plane Constraints: C * q >= d
        self.p_C = cp.Parameter((self.MAX_CONSTRAINTS, self.model.nq), name="C_mat")
        self.p_d = cp.Parameter((self.MAX_CONSTRAINTS,), name="d_vec")

        self.X = cp.Variable((self.nx, self.N+1))
        self.U = cp.Variable((self.nu, self.N))
        
        cost = 0
        constraints = [self.X[:, 0] == self.p_x0]
        
        Q_sqrt = np.sqrt(self.Q_diag)
        R_sqrt = np.sqrt(self.R_diag)

        for k in range(self.N):
            state_error = self.X[:, k] - self.p_xref
            cost += cp.sum_squares(cp.multiply(Q_sqrt, state_error)) 
            cost += cp.sum_squares(cp.multiply(R_sqrt, self.U[:, k]))
            
            # Dynamics
            constraints += [self.X[:, k+1] == self.p_Ad @ self.X[:, k] + self.p_Bd @ self.U[:, k] + self.p_dd]
            
            # Input Limits
            constraints += [self.U[:, k] <= [500]*self.nu]
            constraints += [self.U[:, k] >= [-500]*self.nu]

            # --- WALL BOUNDS ---
            constraints += [self.X[0, k+1] >= self.bounds['x_min']]
            constraints += [self.X[0, k+1] <= self.bounds['x_max']]
            constraints += [self.X[1, k+1] >= self.bounds['y_min']]
            constraints += [self.X[1, k+1] <= self.bounds['y_max']]

            # --- TANGENT PLANE CONSTRAINTS ---
            q_next = self.X[:self.model.nq, k+1]
            constraints += [self.p_C @ q_next >= self.p_d]

        term_error = self.X[:, self.N] - self.p_xref
        cost += cp.quad_form(term_error, cp.psd_wrap(self.P_val))

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def get_control_action(self, q, v, u_last):
        Ac, Bc, d = get_linear_dynamics(q, v, u_last, self.model, self.data)
        Ad, Bd, dd = discretize_dynamics(Ac, Bc, d, self.dt)
        
        C_vals, d_vals = self._get_tangent_plane_constraints(q)
        
        self.p_Ad.value = Ad; self.p_Bd.value = Bd; self.p_dd.value = dd
        self.p_x0.value = np.concatenate([q, v])
        self.p_xref.value = self.x_ref_val
        self.p_C.value = C_vals
        self.p_d.value = d_vals
        
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-3, eps_rel=1e-3)
        except Exception:
            return np.zeros(self.nu), None

        if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return np.zeros(self.nu), None
            
        return self.U[:, 0].value, self.X.value

# ==========================================
# 3. VISUALIZER
# ==========================================

class MPCVisualizer:
    def __init__(self, p_client):
        self.p = p_client
        self.line_ids = []
        self.prev_horizon = 0

    def draw_trajectory(self, X_pred):
        if X_pred is None: return
        N = X_pred.shape[1] - 1
        
        if len(self.line_ids) != N:
            for line_id in self.line_ids:
                self.p.removeUserDebugItem(line_id)
            self.line_ids = [-1] * N 

        for k in range(N):
            start = [X_pred[0, k],   X_pred[1, k],   0.02]
            end   = [X_pred[0, k+1], X_pred[1, k+1], 0.02]
            
            if self.line_ids[k] == -1:
                self.line_ids[k] = self.p.addUserDebugLine(start, end, [0, 1, 0], 3, 0)
            else:
                self.p.addUserDebugLine(start, end, [0, 1, 0], 3, 0, replaceItemUniqueId=self.line_ids[k])