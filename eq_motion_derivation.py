import pinocchio as pin
import numpy as np
import os

# Save your XML string to a file for Pinocchio to read
# (Ensure you save the XML content you provided to this file)


def get_linear_dynamics(q_current, v_current, u_current, model, data):
    """
    Computes dx = A*x + B*u + d
    Linearized around (q, v, u_current)
    """
    n_v = model.nv

    # 1. Compute Derivatives (A and B parts)
    # This fills data.ddq_dq, data.ddq_dv, data.Minv
    pin.computeABADerivatives(model, data, q_current, v_current, u_current)
    
    # 2. Compute Nominal Dynamics f(x0, u0)
    # This computes purely the acceleration at the current state
    # Result is stored in data.ddq
    pin.aba(model, data, q_current, v_current, u_current)
    acc_nominal = data.ddq

    # 3. Construct Continuous Matrices
    Ac = np.zeros((2*n_v, 2*n_v))
    Bc = np.zeros((2*n_v, n_v))
    
    # Kinematics
    Ac[:n_v, n_v:] = np.eye(n_v)
    
    # Dynamics Gradients
    Ac[n_v:, :n_v] = data.ddq_dq
    Ac[n_v:, n_v:] = data.ddq_dv
    Bc[n_v:, :]    = data.Minv
    
    # 4. Compute the Affine "Drift" Term d
    # d = f(x0, u0) - A*x0 - B*u0
    # Note: x0 is [q; v]
    x0 = np.concatenate([q_current, v_current])
    f_x0_u0 = np.concatenate([v_current, acc_nominal]) # State derivative [v; a]
    
    d = f_x0_u0 - Ac @ x0 - Bc @ u_current
    
    return Ac, Bc, d

#q_current = np.zeros(5) # Current Position
#v_current = np.zeros(5) # Current Velocity
#u_current = np.zeros(5) # Current Control Input (Torque)

#model = pin.buildModelFromUrdf("URDF/mobileManipulator.urdf")
#data = model.createData()

#print(get_linear_dynamics(q_current, v_current, u_current, model, data))

def get_nonlinear_dynamics():
    # Load model (as done previously)
    model = pin.buildModelFromUrdf("URDF/mobileManipulator.urdf" )
    data = model.createData()

    q = np.zeros(model.nq) # Current Position
    v = np.zeros(model.nv) # Current Velocity
    a_zero = np.zeros(model.nv) # Zero acceleration

    # --- 1. Get Mass Matrix M(q) ---
    # Use CRBA (Composite Rigid Body Algorithm)
    # Note: Pinocchio stores the upper triangle. use full() to get the symmetric matrix
    pin.crba(model, data, q)
    M = data.M 
    # Include the motors' rotor inertia if you have them, usually handled in URDF
    # To ensure it's symmetric in standard numpy usage:
    M = np.triu(M) + np.triu(M, 1).T 

    # --- 2. Get Nonlinear Bias h(q, v) = C*v + g ---
    # Use RNEA (Recursive Newton-Euler Algorithm) with acceleration = 0
    # "If I want 0 acceleration, what torque do I need to hold the robot 
    # against gravity and coriolis?" -> That IS the bias term.
    h = pin.rnea(model, data, q, v, a_zero)

    # --- 3. Get Gravity only g(q) ---
    # Use RNEA with velocity=0 and acceleration=0
    g = pin.computeGeneralizedGravity(model, data, q)

    # --- 4. Get Coriolis only C(q, v)v ---
    # Simply subtract gravity from the bias
    C_v = h - g
    return M, h, C_v, g
