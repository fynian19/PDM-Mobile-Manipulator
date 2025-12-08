from eq_motion_derivation import get_linear_dynamics
import numpy as np
import pinocchio as pin

class MPCController:
    def __init__(self, urdf_path):
        # Initialize Pinocchio model
        self.model = pin.buildModelFromUrdf(urdf_path)
        
        self.data = self.model.createData() 
        
        # Pre-allocate reuseable arrays for the solver to save time
        self.dt = 0.05

    def get_control_action(self, q, v, u_last):
        # --- FAST PART (Run at 50Hz) ---
        
        # 1. We pass 'self.data' into the function.
        # The function overwrites the numbers inside 'self.data' with new ones.
        # It does NOT create a new object.
        Ac, Bc, d = get_affine_dynamics(self.model, self.data, q, v, u_last)
        
        # 2. Discretize
        Ad = np.eye(self.model.nv * 2) + Ac * self.dt
        Bd = Bc * self.dt
        dd = d * self.dt
        
        # 3. Solve MPC...
        return optimal_u
    
    def solve_mpc(self):
        u = np.ones(self.model.nv)*0  # Placeholder
        u[2] = -80
        return u