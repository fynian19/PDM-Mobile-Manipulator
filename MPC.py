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

    def __init__(self, urdf_path, x_ref, dt, N=10, obstacle_params=None):
        self.dt = dt
        self.N = N 
        self.x_ref_val = x_ref
        
        # 1. Setup Pinocchio
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData() 
        self.nx = self.model.nv * 2
        self.nu = self.model.nv

        self.x_ref_val = x_ref
        self.obstacle_params = obstacle_params # Store obstacle info

        # 2. Weights (Store as numpy arrays for math)
        self.Q_diag = np.array([100, 100, 50, 50, 50] + [10, 10, 5, 5, 5]) 
        self.R_diag = np.array([0.1] * self.nu) # Fixed your R shape logic
        
        # 3. Compute Terminal Cost P
        self._compute_terminal_cost()

        # 4. BUILD CVXPY PROBLEM ONCE (The Speed Fix)
        self._setup_mpc_problem()

    def _compute_terminal_cost(self):
        # Linearize at the goal (x_ref) with 0 velocity
        q_goal = self.x_ref_val[:self.model.nq]
        v_goal = np.zeros(self.model.nv)
        
        # Get Dynamics at Goal
        Ac, Bc, _ = get_linear_dynamics(q_goal, v_goal, np.zeros(self.nu), self.model, self.data)
        
        # Discretize
        Ad = np.eye(self.nx) + Ac * self.dt
        Bd = Bc * self.dt
        
        # Solve DARE
        Q_mat = np.diag(self.Q_diag)
        R_mat = np.diag(self.R_diag)
        self.P_val = scipy.linalg.solve_discrete_are(Ad, Bd, Q_mat, R_mat)

    def _compute_obstacle_plane(self, q_current):
        """
        Calculates the separating plane (Normal n, Distance d)
        Constraint: n.T @ [x,y] >= d
        """
        if self.obstacle_params is None:
            # No obstacle: Return dummy constraint (0 >= -Inf)
            return np.zeros(2), np.array([-1000.0])

        # 1. Extract Positions
        robot_pos = q_current[:2] # x, y
        obs_pos = self.obstacle_params[0,:2] # x, y
        obs_rad = self.obstacle_params[0,3]
        
        # 2. Compute Normal Vector (from Obstacle -> Robot)
        diff = robot_pos - obs_pos
        dist = np.linalg.norm(diff)
        
        # Safety margin (Robot radius approx 0.5m)
        safety_margin = 0.6 
        
        if dist < 1e-3:
            # Singularity (Robot inside obstacle center) - Push X+
            normal = np.array([1.0, 0.0])
        else:
            normal = diff / dist
            
        # 3. Compute Distance Limit (d)
        # The wall is located at: Obstacle_Center + Normal * (Radius + Margin)
        # d = normal . point_on_wall
        point_on_wall = obs_pos + normal * (obs_rad + safety_margin)
        d_val = np.dot(normal, point_on_wall)
        
        return normal, np.array([d_val])

    def _setup_mpc_problem(self):
        """
        Defines the MPC problem symbolically ONCE.
        We use cp.Parameter for matrices that change every step.
        """
        # --- Parameters (Placeholders for data) ---
        self.p_Ad = cp.Parameter((self.nx, self.nx), name="Ad")
        self.p_Bd = cp.Parameter((self.nx, self.nu), name="Bd")
        self.p_dd = cp.Parameter((self.nx,), name="dd")
        self.p_x0 = cp.Parameter((self.nx,), name="x0")
        self.p_xref = cp.Parameter((self.nx,), name="xref")
        
        # --- Variables ---
        self.X = cp.Variable((self.nx, self.N+1))
        self.U = cp.Variable((self.nu, self.N))
        
        # --- Cost & Constraints ---
        cost = 0
        constraints = [self.X[:, 0] == self.p_x0]

        # --- NEW: Obstacle Constraint Parameters ---
        self.p_obs_n = cp.Parameter((2,), name="obs_n") # Normal vector
        self.p_obs_d = cp.Parameter((1,), name="obs_d") # Scalar distance

        # Pre-compute sqrt matrices for efficient 'sum_squares' (Standard CVXPY trick)
        Q_sqrt = np.sqrt(self.Q_diag)
        R_sqrt = np.sqrt(self.R_diag)

        for k in range(self.N):
            # Cost (Tracking + Effort)
            state_error = self.X[:, k] - self.p_xref
            cost += cp.sum_squares(cp.multiply(Q_sqrt, state_error)) 
            cost += cp.sum_squares(cp.multiply(R_sqrt, self.U[:, k]))
            
            # Dynamics Update
            constraints += [self.X[:, k+1] == self.p_Ad @ self.X[:, k] + self.p_Bd @ self.U[:, k] + self.p_dd]
            
            # Constraints (Make sure limits match your URDF)
            constraints += [self.U[:, k] <= [30, 30, 30, 10, 10]]
            constraints += [self.U[:, k] >= [-30, -30, -30, -10, -10]]

            # --- OBSTACLE CONSTRAINT ---
            # n.T * pos >= d
            constraints += [self.p_obs_n @ self.X[:2, k+1] >= self.p_obs_d]
            
        # Terminal Cost (Using P)
        term_error = self.X[:, self.N] - self.p_xref
        cost += cp.quad_form(term_error, cp.psd_wrap(self.P_val))

        # Compile!
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def get_control_action(self, q, v, u_last):
        # 1. Linearize Dynamics (Fast)
        Ac, Bc, d = get_linear_dynamics(q, v, u_last, self.model, self.data)
        Ad, Bd, dd = discretize_dynamics(Ac, Bc, d, self.dt)
        
        # 2. Update Parameters (No compilation needed!)
        self.p_Ad.value = Ad
        self.p_Bd.value = Bd
        self.p_dd.value = dd
        self.p_x0.value = np.concatenate([q, v])
        self.p_xref.value = self.x_ref_val
        
        # Update Obstacle Params
        n_val, d_val = self._compute_obstacle_plane(q)
        self.p_obs_n.value = n_val
        self.p_obs_d.value = d_val

        try:
            self.prob.solve(
                solver=cp.OSQP, 
                warm_start=True,
                eps_abs=1e-3, # Looser tolerance (was default 1e-5)
                eps_rel=1e-3, 
                max_iter=1000 # Give it more time if needed
            )
        except Exception as e:
            print(f"Solver crashed: {e}")
            # FIX 1: Return a tuple (u, X) even on crash
            return np.zeros(self.nu), None

        # FIX 2: Accept "optimal_inaccurate" as success
        # It just means the solver reached the limit of floating point precision
        if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Solver failed: {self.prob.status}")
            # FIX 3: Return tuple on feasibility failure
            return np.zeros(self.nu), None
            
        # Success path
        return self.U[:, 0].value, self.X.value
    
    def dummy_mpc(self):
        u = np.ones(self.model.nv)*0  # Placeholder
        u[2] = -80
        return u
    

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
