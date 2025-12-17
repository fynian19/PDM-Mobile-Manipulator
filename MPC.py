from eq_motion_derivation import discretize_dynamics, get_linear_dynamics
import numpy as np
import pinocchio as pin
import cvxpy as cp
import scipy


def get_clamped_reference(x_current, x_global, max_lookahead=3.0):
    """
    Creates a temporary local target 'max_lookahead' meters away 
    in the direction of the global target.
    
    x_current: [x, y, theta, ...]
    x_global:  [x_goal, y_goal, theta_goal, ...]
    """
    # Extract positions (Base X, Base Y)
    curr_pos = x_current[:2]
    glob_pos = x_global[:2]
    
    # Compute vector to target
    diff = glob_pos - curr_pos
    dist = np.linalg.norm(diff)
    
    # If we are far away, clamp the target
    if dist > max_lookahead:
        # Normalize direction and scale by lookahead distance
        direction = diff / dist
        local_pos = curr_pos + direction * max_lookahead
        
        # Create new reference vector
        x_ref_local = x_global.copy()
        x_ref_local[0] = local_pos[0]
        x_ref_local[1] = local_pos[1]
        
        return x_ref_local
    else:
        # We are close enough, go straight to the global target
        return x_global

# ==========================================
# 1. MPC CONTROLLER CLASS
class MPCController:
    def __init__(self, urdf_path, x_ref, dt, N=10, obstacle_list=None):
        self.dt = dt
        self.N = N 
        self.x_ref_val = x_ref
        self.obstacle_list = obstacle_list if obstacle_list else []
        self.MAX_OBS = 3 # Max number of active constraints (keep low for speed)
        
        # Setup Pinocchio
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData() 
        self.nx = self.model.nv * 2
        self.nu = self.model.nv

        # Tuning
        self.Q_diag = np.array([100, 100, 50, 50, 50] + [10, 10, 10, 5, 5]) 
        self.R_diag = np.array([0.5] * self.nu) 
        
        self._compute_terminal_cost()
        self._setup_mpc_problem()

    def _compute_terminal_cost(self):
        q_goal = self.x_ref_val[:self.model.nq]
        v_goal = np.zeros(self.model.nv)
        Ac, Bc, _ = get_linear_dynamics(q_goal, v_goal, np.zeros(self.nu), self.model, self.data)
        Ad = np.eye(self.nx) + Ac * self.dt
        Bd = Bc * self.dt
        Q_mat = np.diag(self.Q_diag)
        R_mat = np.diag(self.R_diag)
        self.P_val = scipy.linalg.solve_discrete_are(Ad, Bd, Q_mat, R_mat)

    def _get_obstacle_constraints(self, q_current):
        """
        Selects the 3 closest obstacles and computes linear constraints.
        """
        robot_pos = q_current[:2]
        robot_radius = 0.6 # Safety margin
        
        candidates = []
        
        for obs in self.obstacle_list:
            obs_pos = np.array(obs['pos'])
            
            if obs['type'] == 'CYL':
                # Cylinder Logic
                diff = robot_pos - obs_pos
                dist_center = np.linalg.norm(diff)
                
                if dist_center < 1e-4: normal = np.array([1.0, 0])
                else: normal = diff / dist_center
                
                # Surface point logic
                d_val = np.dot(normal, obs_pos + normal * (obs['radius'] + robot_radius))
                dist_surface = dist_center - obs['radius']
                
            elif obs['type'] == 'BOX':
                # Box Logic: Find closest point on rectangle
                half_size = np.array(obs['size'])
                # Clamp robot position to the box extent
                closest_x = np.clip(robot_pos[0], obs_pos[0] - half_size[0], obs_pos[0] + half_size[0])
                closest_y = np.clip(robot_pos[1], obs_pos[1] - half_size[1], obs_pos[1] + half_size[1])
                closest_point = np.array([closest_x, closest_y])
                
                diff = robot_pos - closest_point
                dist_surface = np.linalg.norm(diff)
                
                if dist_surface < 1e-4: # Inside box
                    normal = (robot_pos - obs_pos) # Push away from center
                    if np.linalg.norm(normal) == 0: normal = np.array([1.0, 0])
                    else: normal /= np.linalg.norm(normal)
                else:
                    normal = diff / dist_surface
                
                d_val = np.dot(normal, closest_point + normal * robot_radius)

            candidates.append((dist_surface, normal, d_val))

        # Sort by distance and take top MAX_OBS
        candidates.sort(key=lambda x: x[0])
        
        # Initialize with dummy constraints (Always True: 0*x >= -1000)
        n_out = np.zeros((self.MAX_OBS, 2))
        d_out = np.full(self.MAX_OBS, -1000.0) 
        
        for i in range(min(len(candidates), self.MAX_OBS)):
            n_out[i, :] = candidates[i][1]
            d_out[i]    = candidates[i][2]
            
        return n_out, d_out

    def _setup_mpc_problem(self):
        # ... (Standard Parameters) ...
        self.p_Ad = cp.Parameter((self.nx, self.nx), name="Ad")
        self.p_Bd = cp.Parameter((self.nx, self.nu), name="Bd")
        self.p_dd = cp.Parameter((self.nx,), name="dd")
        self.p_x0 = cp.Parameter((self.nx,), name="x0")
        self.p_xref = cp.Parameter((self.nx,), name="xref")
        
        # --- NEW: Multiple Obstacles (Shape is N x 2) ---
        self.p_obs_n = cp.Parameter((self.MAX_OBS, 2), name="obs_n") 
        self.p_obs_d = cp.Parameter((self.MAX_OBS,), name="obs_d")

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
            constraints += [self.U[:, k] <= [500, 500, 500, 200, 200]]
            constraints += [self.U[:, k] >= [-500, -500, -500, -200, -200]]

            # --- MULTI-OBSTACLE CONSTRAINTS ---
            # Matrix multiplication applies all 3 constraints at once
            constraints += [self.p_obs_n @ self.X[:2, k+1] >= self.p_obs_d]

        term_error = self.X[:, self.N] - self.p_xref
        cost += cp.quad_form(term_error, cp.psd_wrap(self.P_val))

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def get_control_action(self, q, v, u_last):
        # Linearize Dynamics
        Ac, Bc, d = get_linear_dynamics(q, v, u_last, self.model, self.data)
        Ad, Bd, dd = discretize_dynamics(Ac, Bc, d, self.dt)
        
        # Linearize Obstacles (Dynamic Selection)
        n_vals, d_vals = self._get_obstacle_constraints(q)

        # Update Params
        self.p_Ad.value = Ad
        self.p_Bd.value = Bd
        self.p_dd.value = dd
        self.p_x0.value = np.concatenate([q, v])
        self.p_xref.value = self.x_ref_val
        self.p_obs_n.value = n_vals
        self.p_obs_d.value = d_vals
        
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-3, eps_rel=1e-3)
        except Exception:
            return np.zeros(self.nu), None

        if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return np.zeros(self.nu), None
            
        return self.U[:, 0].value, self.X.value
    

# ==========================================
# 2. SETUP PYBULLET SIMULATION
# ==========================================

class MPCVisualizer:
    def __init__(self, p_client):
        self.p = p_client
        self.line_ids = []  # Stores the integer IDs of the debug lines
        self.prev_horizon = 0

    def draw_trajectory(self, X_pred):
        """
        Updates the green MPC trajectory lines.
        X_pred: (nx, N+1) array
        """
        if X_pred is None: return

        # Number of segments to draw
        N = X_pred.shape[1] - 1
        
        # 1. If horizon changed (or first run), clear old lines and reset
        if len(self.line_ids) != N:
            for line_id in self.line_ids:
                self.p.removeUserDebugItem(line_id)
            self.line_ids = [-1] * N # Initialize with -1 (meaning "not created yet")

        # 2. Update lines (Move them, don't delete them)
        for k in range(N):
            start = [X_pred[0, k],   X_pred[1, k],   0.02]
            end   = [X_pred[0, k+1], X_pred[1, k+1], 0.02]
            
            if self.line_ids[k] == -1:
                # CREATE for the first time
                self.line_ids[k] = self.p.addUserDebugLine(
                    lineFromXYZ=start,
                    lineToXYZ=end,
                    lineColorRGB=[0, 1, 0],
                    lineWidth=3,
                    lifeTime=0 # 0 means "persist forever" (until we remove it)
                )
            else:
                # UPDATE existing line (Zero overhead)
                self.p.addUserDebugLine(
                    lineFromXYZ=start,
                    lineToXYZ=end,
                    lineColorRGB=[0, 1, 0],
                    lineWidth=3,
                    lifeTime=0,
                    replaceItemUniqueId=self.line_ids[k] # <--- THE KEY FIX
                )
