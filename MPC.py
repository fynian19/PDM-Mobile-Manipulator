from eq_motion_derivation import discretize_dynamics, get_linear_dynamics
import numpy as np
import pinocchio as pin
import cvxpy as cp
import scipy
class MPCController:

    def __init__(self, urdf_path, x_ref, dt, N=20):
        self.dt = dt
        self.N = N 
        self.x_ref_val = x_ref
        
        # 1. Setup Pinocchio
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData() 
        self.nx = self.model.nv * 2
        self.nu = self.model.nv

        # 2. Weights (Store as numpy arrays for math)
        self.Q_diag = np.array([100, 100, 100, 100, 100] + [1, 1, 1, 1, 1]) 
        self.R_diag = np.array([0.00001] * self.nu) # Fixed your R shape logic
        
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
            constraints += [self.U[:, k] <= [50, 50, 50, 10, 10]]
            constraints += [self.U[:, k] >= [-50, -50, -50, -10, -10]]

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
        
        # 3. Solve (Fast)
        try:
            # warm_start=True reuses the previous solution as a guess
            self.prob.solve(solver=cp.OSQP, warm_start=True)
        except Exception as e:
            print(f"Solver failed: {e}")
            return np.zeros(self.nu)

        if self.prob.status != cp.OPTIMAL:
            print(f"Infeasible: {self.prob.status}")
            return np.zeros(self.nu)
            
        # Return first control action
        return self.U[:, 0].value, self.X.value
    
    def dummy_mpc(self):
        u = np.ones(self.model.nv)*0  # Placeholder
        u[2] = -80
        return u
    

# ==========================================
# 2. SETUP PYBULLET SIMULATION
# ==========================================
def draw_mpc_path(p, X_pred, lifetime):
    """
    Draws the predicted MPC trajectory in PyBullet.
    X_pred: (nx, N+1) array. 
            Rows 0,1 must be x,y.
    """
    if X_pred is None: return

    # We iterate through the horizon N
    for k in range(X_pred.shape[1] - 1):
        # Start point (x, y, z=0.02)
        start = [X_pred[0, k], X_pred[1, k], 0.02] 
        # End point
        end   = [X_pred[0, k+1], X_pred[1, k+1], 0.02]
        
        p.addUserDebugLine(
            lineFromXYZ=start,
            lineToXYZ=end,
            lineColorRGB=[0, 1, 0], # Green color
            lineWidth=3,
            lifeTime=lifetime # Disappears automatically after 0.1s
        )