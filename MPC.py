from eq_motion_derivation import discretize_dynamics, get_linear_dynamics
import numpy as np
import pinocchio as pin
import cvxpy as cp
import scipy
class MPCController:
    def __init__(self, urdf_path, x_ref):
        # Initialize Pinocchio model
        self.model = pin.buildModelFromUrdf(urdf_path)
        
        self.data = self.model.createData() 
        
        # Pre-allocate reuseable arrays for the solver to save time
        self.dt = 0.05
        self.x_ref = x_ref

    def solve_mpc_cvxpy(self, Ad, Bd, dd, x0, x_ref, N=10):
        """
        Solves the Linear MPC problem using CVXPY
        """
        nx, nu = Bd.shape
        
        # Variables
        X = cp.Variable((nx, N+1))
        U = cp.Variable((nu, N))
        
        # Cost Weights
        # State: [x, y, theta, q1, q2, vx, vy, vtheta, vq1, vq2]
        # We prioritize reaching the position (first 5)
        Q = np.diag([100, 100, 10, 10, 10] + [1, 1, 1, 1, 1]) 
        R = np.eye(nu) * 0.001  # Low penalty on torque (cheap control)
        P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
        # Solve Discrete-time Algebraic Riccati Equation for terminal cost
        
        cost = 0
        constraints = [X[:, 0] == x0]
        
        for k in range(N):
            #print(X[:, k], x_ref)
            cost += cp.quad_form(X[:, k] - x_ref, Q) + cp.quad_form(U[:, k], R)
            constraints += [X[:, k+1] == Ad @ X[:, k] + Bd @ U[:, k] + dd]
            
            # Input limits (approximate based on URDF)
            constraints += [U[:, k] <= [500, 500, 80, 50, 50]]
            constraints += [U[:, k] >= [-500, -500, -80, -50, -50]]

        # Terminal Cost
        cost += cp.quad_form(X[:, N] - x_ref, P)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        
        if prob.status != cp.OPTIMAL:
            print("MPC Infeasible!")
            return np.zeros(nu)
            
        return U[:, 0].value
    
    
    def get_control_action(self, q, v, u_last):
        # --- FAST PART (Run at 50Hz) ---
        
        # 1. We pass 'self.data' into the function.
        # The function overwrites the numbers inside 'self.data' with new ones.
        # It does NOT create a new object.
        Ac, Bc, d = get_linear_dynamics(q, v, u_last, self.model, self.data)
        Ad, Bd, dd = discretize_dynamics(Ac, Bc, d, self.dt)
        optimal_u = self.solve_mpc_cvxpy(Ad, Bd, dd, np.concatenate([q, v]), self.x_ref)
        print("Optimal u:", optimal_u)
        return optimal_u
    
    def dummy_mpc(self):
        u = np.ones(self.model.nv)*0  # Placeholder
        u[2] = -80
        return u
    
    