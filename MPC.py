import numpy as np
import pinocchio as pin
import cvxpy as cp
import scipy.linalg
from eq_motion_derivation import discretize_dynamics, get_linear_dynamics

# ==========================================
# DISTANCE HELPER
# ==========================================

def update_ee_distance(p_client, robot_id, link_idx, prev_pos, current_total_dist):
    """
    Computes the distance traveled by a specific link since the last update.
    
    Args:
        p_client: The pybullet instance (p).
        robot_id: The ID of the robot body.
        link_idx: The index of the link to track.
        prev_pos: The (x,y,z) position from the previous step (or None).
        current_total_dist: The accumulated distance so far.
        
    Returns:
        (new_pos, updated_total_dist): Tuple to update your state variables.
    """
    # Get current link state (Index 0 is the center of mass position)
    state = p_client.getLinkState(robot_id, link_idx)
    curr_pos = np.array(state[0])

    # Case 1: First step (no previous position)
    if prev_pos is None:
        return curr_pos, current_total_dist

    # Case 2: Calculate Euclidean distance
    dist_step = np.linalg.norm(curr_pos - prev_pos)

    # Filter noise: Ignore microscopic movements (jitter) < 0.1mm
    if dist_step > 1e-4:
        current_total_dist += dist_step

    return curr_pos, current_total_dist

# ==========================================
# GEOMETRY HELPER
# ==========================================
def get_base_facing_plane(p_base, a, b, radius):
    ab = b - a; ap = p_base - a
    denom = np.dot(ab, ab)
    t = np.dot(ap, ab) / denom if denom > 1e-6 else 0.0
    t = np.clip(t, 0.0, 1.0)
    closest_spine = a + t * ab
    diff = p_base - closest_spine
    dist_center = np.linalg.norm(diff)
    if dist_center < 1e-5: n_base = np.array([1.0, 0, 0])
    else: n_base = diff / dist_center
    surface_point = closest_spine + n_base * radius
    return surface_point, n_base

def get_clamped_reference(x_current, x_global, max_lookahead=3.0):
    curr_pos = x_current[:2]; glob_pos = x_global[:2]
    diff = glob_pos - curr_pos
    dist = np.linalg.norm(diff)
    if dist > max_lookahead:
        direction = diff / dist
        local_pos = curr_pos + direction * max_lookahead
        x_ref_local = x_global.copy()
        x_ref_local[0] = local_pos[0]; x_ref_local[1] = local_pos[1]
        return x_ref_local
    return x_global

# ==========================================
# 2. MPC CONTROLLER CLASS
# ==========================================
class MPCController:
    def __init__(self, urdf_path, x_ref, dt, N=10, obstacle_list=None, bounds=None):
        self.dt = dt; self.N = N; self.x_ref_val = x_ref
        # Default to empty list if None
        self.obstacle_list = obstacle_list if obstacle_list else []
        self.bounds = bounds if bounds else {'x_min': -10, 'x_max': 10, 'y_min': -10, 'y_max': 10}
        
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData() 
        self.nx = self.model.nv * 2; self.nu = self.model.nv

        # Weights
        self.Q_diag = np.array([1000, 1000, 100, 100, 100] + [10]*5) 
        self.R_diag = np.array([1.0] * self.nu) 
        
        # Frame Selection
        self.base_frames = []
        self.arm_frames = []
        base_id = self.model.getFrameId("base_link")
        if base_id < self.model.nframes: self.base_frames.append(base_id)
        if not self.base_frames and self.model.nframes > 1: self.base_frames.append(1)

        for i, f in enumerate(self.model.frames):
            if "link" in f.name and "base" not in f.name:
                self.arm_frames.append(i)
                
        self.MAX_CONSTRAINTS = 10
        self._compute_terminal_cost()
        self._setup_mpc_problem()

    def update_weights(self, q_weights):
        self.Q_diag = q_weights
        self._compute_terminal_cost()
        self._setup_mpc_problem()
        
    def update_obstacles(self, obs_list):
        self.obstacle_list = obs_list

    def _compute_terminal_cost(self):
        q_goal = np.zeros(self.model.nq); v_goal = np.zeros(self.model.nv)
        if self.model.nq > self.model.nv: q_goal[2] = 1.0 
        
        try:
            Ac, Bc, _ = get_linear_dynamics(q_goal, v_goal, np.zeros(self.nu), self.model, self.data)
            Ac = Ac[:self.nx, :self.nx]; Bc = Bc[:self.nx, :self.nu]
        except:
            Ac = np.eye(self.nx); Bc = np.zeros((self.nx, self.nu))

        Ad = np.eye(self.nx) + Ac * self.dt; Bd = Bc * self.dt
        self.P_val = scipy.linalg.solve_discrete_are(Ad, Bd, np.diag(self.Q_diag), np.diag(self.R_diag))

    def _get_constraints(self, q_current):
        if self.model.nq > self.model.nv:
            q_pin = np.concatenate([q_current[:2], [np.cos(q_current[2]), np.sin(q_current[2])], q_current[3:]])
        else: q_pin = q_current

        pin.forwardKinematics(self.model, self.data, q_pin)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q_pin)

        p_base = np.array([q_current[0], q_current[1], 0.3]) 
        pedestal_pos = np.array([0.0, 8.0, 0.25]) 
        
        # Use a single list to keep math and visuals synchronized
        combined_data = []

        for obs in self.obstacle_list:
            raw_pos = np.array(obs.get('pos', [0,0,0]))
            if len(raw_pos)==2: raw_pos = np.append(raw_pos, 0.0)
            
            is_pedestal = (np.linalg.norm(raw_pos - pedestal_pos) < 0.5)
            frames_to_check = self.base_frames if is_pedestal else (self.base_frames + self.arm_frames)

            # Geometry
            if obs['type'] == 'CYL':
                if 'p0' in obs: p0=np.array(obs['p0']); p1=np.array(obs['p1'])
                else: c=raw_pos.copy(); h=obs.get('height', 2.0); p0=c.copy(); p0[2]-=h/2; p1=c.copy(); p1[2]+=h/2
                if len(p0)==2: p0=np.append(p0,0); 
                if len(p1)==2: p1=np.append(p1,1)
                radius = obs.get('radius', 0.5)
            elif obs['type'] == 'BOX':
                size = np.array(obs.get('size', [0.5,0.5,1.0]))
                if np.max(size[:2]) > 4.0: continue 
                if len(size)==2: size=np.append(size, 1.0)
                h = size[2]*2; p0=raw_pos.copy(); p0[2]-=h/2; p1=raw_pos.copy(); p1[2]+=h/2
                radius = np.linalg.norm(size[:2])

            surface_anchor, n_base = get_base_facing_plane(p_base, p0, p1, radius)
            
            # Global Filter
            if np.linalg.norm(p_base - surface_anchor) > 3.0: continue

            for frame_idx in frames_to_check:
                p_link = self.data.oMf[frame_idx].translation
                
                # Robust Euclidean Check
                real_dist = np.linalg.norm(p_link - surface_anchor)
                if real_dist > 1.0: continue 

                dist_proj = np.dot(n_base, p_link - surface_anchor)
                J_lin = pin.getFrameJacobian(self.model, self.data, frame_idx, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
                
                margin = 0.3
                C_row = n_base @ J_lin
                d_val = margin - dist_proj + np.dot(C_row, q_current)
                
                # Store together
                vis_tuple = (dist_proj, C_row, d_val, p_link, n_base)
                combined_data.append( (dist_proj, C_row, d_val, vis_tuple) )

        # Sort by distance (closest first)
        combined_data.sort(key=lambda x: x[0])
        
        C_out = np.zeros((self.MAX_CONSTRAINTS, self.nx // 2))
        d_out = np.full(self.MAX_CONSTRAINTS, -1000.0)
        vis_data = []
        
        count = min(len(combined_data), self.MAX_CONSTRAINTS)
        for i in range(count):
            dist, C, d, v_tup = combined_data[i]
            C_out[i,:] = C
            d_out[i] = d
            vis_data.append(v_tup)
            
        return C_out, d_out, vis_data

    def _setup_mpc_problem(self):
        self.p_Ad = cp.Parameter((self.nx, self.nx)); self.p_Bd = cp.Parameter((self.nx, self.nu))
        self.p_dd = cp.Parameter((self.nx,)); self.p_x0 = cp.Parameter((self.nx,)); self.p_xref = cp.Parameter((self.nx,))
        self.p_C = cp.Parameter((self.MAX_CONSTRAINTS, self.nx // 2)); self.p_d = cp.Parameter((self.MAX_CONSTRAINTS,))

        self.X = cp.Variable((self.nx, self.N+1)); self.U = cp.Variable((self.nu, self.N))
        self.Slack = cp.Variable((self.MAX_CONSTRAINTS, self.N), nonneg=True) 

        cost = 0; constraints = [self.X[:, 0] == self.p_x0]
        Q_sqrt = np.sqrt(self.Q_diag); R_sqrt = np.sqrt(self.R_diag)

        for k in range(self.N):
            cost += cp.sum_squares(cp.multiply(Q_sqrt, self.X[:, k] - self.p_xref)) 
            cost += cp.sum_squares(cp.multiply(R_sqrt, self.U[:, k]))
            cost += 100000 * cp.sum(self.Slack[:, k])
            
            constraints += [self.X[:, k+1] == self.p_Ad @ self.X[:, k] + self.p_Bd @ self.U[:, k] + self.p_dd]
            constraints += [self.U[:, k] <= 300, self.U[:, k] >= -300]
            constraints += [self.X[0, k+1] >= self.bounds['x_min'], self.X[0, k+1] <= self.bounds['x_max']]
            constraints += [self.X[1, k+1] >= self.bounds['y_min'], self.X[1, k+1] <= self.bounds['y_max']]
            constraints += [self.p_C @ self.X[:self.nx//2, k+1] >= self.p_d - self.Slack[:, k]]

        cost += cp.quad_form(self.X[:, self.N] - self.p_xref, cp.psd_wrap(self.P_val))
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def get_control_action(self, q, v, u_last):
        q_pin = q
        if self.model.nq > self.model.nv:
             q_pin = np.concatenate([q[:2], [np.cos(q[2]), np.sin(q[2])], q[3:]])

        Ac, Bc, d = get_linear_dynamics(q_pin, v, u_last, self.model, self.data)
        Ac = Ac[:self.nx, :self.nx]; Bc = Bc[:self.nx, :self.nu]; d = d[:self.nx]
        
        Ad, Bd, dd = discretize_dynamics(Ac, Bc, d, self.dt)
        C_vals, d_vals, vis_data = self._get_constraints(q)
        
        self.p_Ad.value = Ad; self.p_Bd.value = Bd; self.p_dd.value = dd
        self.p_x0.value = np.concatenate([q, v])
        
        ref = self.x_ref_val
        if len(ref) > self.nx: ref = ref[:self.nx]
        self.p_xref.value = ref
        
        self.p_C.value = C_vals; self.p_d.value = d_vals
        
        try: self.prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-3, eps_rel=1e-3)
        except: return np.zeros(self.nu), None, vis_data # RETURN VIS DATA ANYWAY
        
        if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]: 
            # RETURN VIS DATA ANYWAY
            return np.zeros(self.nu), None, vis_data 
            
        return self.U[:, 0].value, self.X.value, vis_data


class MPCVisualizer:
    def __init__(self, p_client):
        self.p = p_client
        self.traj_ids = []
        self.plane_ids = []

    def draw_trajectory(self, X_pred):
        if X_pred is None: return
        N = X_pred.shape[1] - 1
        while len(self.traj_ids) < N: self.traj_ids.append(-1)
        for k in range(N):
            start = [X_pred[0, k], X_pred[1, k], 0.02]; end = [X_pred[0, k+1], X_pred[1, k+1], 0.02]
            if self.traj_ids[k] == -1: self.traj_ids[k] = self.p.addUserDebugLine(start, end, [0,1,0], 3)
            else: self.p.addUserDebugLine(start, end, [0,1,0], 3, replaceItemUniqueId=self.traj_ids[k])

    def draw_planes(self, vis_data):
        if vis_data is None: return
        needed = len(vis_data) * 5
        while len(self.plane_ids) < needed: self.plane_ids.append(-1)
        # Hide unused
        for i in range(needed, len(self.plane_ids)):
            if self.plane_ids[i] != -1: self.p.addUserDebugLine([0,0,0],[0,0,0],[0,0,0],1,replaceItemUniqueId=self.plane_ids[i])

        idx = 0
        for item in vis_data:
            dist, _, _, p_robot, normal = item
            center = p_robot - normal * dist
            
            arb = np.array([0,0,1]) if abs(normal[2]) < 0.9 else np.array([1,0,0])
            t1 = np.cross(normal, arb); t1 /= np.linalg.norm(t1)
            t2 = np.cross(normal, t1)
            s = 0.20 
            pts = [center + t1*s + t2*s, center - t1*s + t2*s, center - t1*s - t2*s, center + t1*s - t2*s]
            col = [1,0,0] if dist < 0.1 else [0,1,1]
            
            for i in range(4):
                uid = self.plane_ids[idx]
                if uid == -1: self.plane_ids[idx] = self.p.addUserDebugLine(pts[i], pts[(i+1)%4], col, 2)
                else: self.p.addUserDebugLine(pts[i], pts[(i+1)%4], col, 2, replaceItemUniqueId=uid)
                idx += 1
            
            uid = self.plane_ids[idx]
            if uid == -1: self.plane_ids[idx] = self.p.addUserDebugLine(center, p_robot, [1,1,0], 1)
            else: self.p.addUserDebugLine(center, p_robot, [1,1,0], 1, replaceItemUniqueId=uid)
            idx += 1