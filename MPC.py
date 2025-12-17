import numpy as np
import pinocchio as pin
import cvxpy as cp
import scipy.linalg
from eq_motion_derivation import discretize_dynamics, get_linear_dynamics

# ==========================================
# 1. GEOMETRY HELPERS
# ==========================================

def get_transform(pos, quat):
    T = np.eye(4)
    T[:3, :3] = pin.Quaternion(np.array(quat)).matrix()
    if len(pos) == 2: T[:2, 3] = pos
    else: T[:3, 3] = pos
    return T

def dist_point_to_segment(p, a, b, radius):
    """
    Computes distance and standard surface normal from segment [a,b] to point p.
    """
    ab = b - a
    ap = p - a
    
    denom = np.dot(ab, ab)
    if denom < 1e-6: t = 0.0
    else: t = np.dot(ap, ab) / denom
    
    t_clamped = np.clip(t, 0.0, 1.0)
    closest_point = a + t_clamped * ab
    
    diff = p - closest_point
    dist_center = np.linalg.norm(diff)
    
    if dist_center < 1e-4:
        normal = np.array([0, 0, 1.0]) 
    else:
        normal = diff / dist_center 
        
    dist_surface = dist_center - radius
    return dist_surface, normal

def dist_point_to_box(point, box_pos, box_quat, box_size):
    if len(box_pos) == 2: box_pos = np.append(box_pos, 0.0)
    if len(box_size) == 2: box_size = np.append(box_size, 1.0)
    
    T_world_box = get_transform(box_pos, box_quat)
    p_local = T_world_box[:3, :3].T @ (point - T_world_box[:3, 3])
    
    closest_local = np.clip(p_local, -np.array(box_size), np.array(box_size))
    
    diff_local = p_local - closest_local
    dist = np.linalg.norm(diff_local)
    
    if dist < 1e-6:
        normal_local = np.array([1.0, 0, 0])
        dist = -1e-4
    else:
        normal_local = diff_local / dist

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

        # Tuning
        self.Q_diag = np.array([100, 100, 50, 50, 50] + [10, 10, 10, 5, 5]) 
        self.R_diag = np.array([0.5] * self.nu) 
        
        # --- COLLISION CHECK POINTS ---
        self.collision_frames = []
        
        # Try to find base_link ID safely
        base_id = self.model.getFrameId("base_link")
        if base_id < self.model.nframes:
            self.collision_frames.append(base_id)
        
        for i, f in enumerate(self.model.frames):
            if "link" in f.name and "base" not in f.name:
                self.collision_frames.append(i)
        
        self.MAX_CONSTRAINTS = 15 
        
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
        Generates constraints.
        Normal vector points towards Base Center (X,Y) if safe.
        """
        pin.forwardKinematics(self.model, self.data, q_current)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_current)

        # Get current Base Position (X, Y)
        # Note: q_current[0] is X, q_current[1] is Y
        base_center_xy = q_current[:2]

        candidates = []

        for frame_idx in self.collision_frames:
            p_robot = self.data.oMf[frame_idx].translation
            J_lin = pin.getFrameJacobian(self.model, self.data, frame_idx, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :] 

            for obs in self.obstacle_list:
                
                # 1. Get Geometry Data (Dist + Surface Normal)
                if obs['type'] == 'CYL':
                    p0 = np.array(obs.get('p0', [0,0,0])); p1 = np.array(obs.get('p1', [0,0,1]))
                    if len(p0)==2: p0=np.append(p0,0); 
                    if len(p1)==2: p1=np.append(p1,1)
                    radius = obs.get('radius', 0.5)
                    dist, n_surf = dist_point_to_segment(p_robot, p0, p1, radius)
                
                elif obs['type'] == 'BOX':
                    pos = np.array(obs.get('pos', [0,0,0])); quat = obs.get('quat', [0,0,0,1])
                    size = np.array(obs.get('size', [0.1,0.1,0.1]))
                    if len(pos)==2: pos=np.append(pos,0)
                    if len(size)==2: size=np.append(size,1)
                    dist, n_surf = dist_point_to_box(p_robot, pos, quat, size)

                # 2. CALCULATE "TO BASE" NORMAL
                # Vector from Joint to Base Center (in 3D, assuming Base Z=0 or same as Joint Z for simplicity)
                # Ideally, we want to push the joint horizontally towards the base.
                vec_to_base = np.array([base_center_xy[0], base_center_xy[1], p_robot[2]]) - p_robot
                
                # Normalize
                norm_base = np.linalg.norm(vec_to_base)
                if norm_base > 1e-4:
                    n_base = vec_to_base / norm_base
                else:
                    n_base = n_surf # Fallback if at center
                
                # 3. SAFETY CHECK (Dot Product)
                # Does pushing towards the base actually move us away from the obstacle?
                # Check alignment with the Surface Normal (n_surf).
                # n_surf points OUT of obstacle.
                alignment = np.dot(n_base, n_surf)
                
                if alignment > 0.1: 
                    # Yes, pushing to base moves us away from obstacle. Use n_base.
                    final_normal = n_base
                else:
                    # No, Base is "behind" the wall or perpendicular. Pushing there crashes.
                    # Fallback to standard surface normal to ensure safety.
                    final_normal = n_surf

                # 4. Formulate Constraint
                margin = 0.2 
                C_row = final_normal @ J_lin
                
                if dist < margin:
                    req_dist = dist + 0.05 
                else:
                    req_dist = margin
                
                # Constraint: C * q_next >= req - dist + C * q_curr
                d_val = req_dist - dist + np.dot(C_row, q_current)
                
                candidates.append((dist, C_row, d_val))

        candidates.sort(key=lambda x: x[0])
        
        C_out = np.zeros((self.MAX_CONSTRAINTS, self.model.nq))
        d_out = np.full(self.MAX_CONSTRAINTS, -1000.0) 
        
        for i in range(min(len(candidates), self.MAX_CONSTRAINTS)):
            C_out[i, :] = candidates[i][1]
            d_out[i]    = candidates[i][2]
            
        return C_out, d_out

    def _setup_mpc_problem(self):
        self.p_Ad = cp.Parameter((self.nx, self.nx), name="Ad")
        self.p_Bd = cp.Parameter((self.nx, self.nu), name="Bd")
        self.p_dd = cp.Parameter((self.nx,), name="dd")
        self.p_x0 = cp.Parameter((self.nx,), name="x0")
        self.p_xref = cp.Parameter((self.nx,), name="xref")
        
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
            
            constraints += [self.X[:, k+1] == self.p_Ad @ self.X[:, k] + self.p_Bd @ self.U[:, k] + self.p_dd]
            constraints += [self.U[:, k] <= [500]*self.nu]
            constraints += [self.U[:, k] >= [-500]*self.nu]

            constraints += [self.X[0, k+1] >= self.bounds['x_min']]
            constraints += [self.X[0, k+1] <= self.bounds['x_max']]
            constraints += [self.X[1, k+1] >= self.bounds['y_min']]
            constraints += [self.X[1, k+1] <= self.bounds['y_max']]

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