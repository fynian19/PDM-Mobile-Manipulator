import numpy as np
from pyroboplan.planning.utils import discretize_joint_space_path


def FK(l1, l2, l3, l4, x_b, y_b, q1, q2, q3):
    # l4 is just attached rigidly to l3, such that the upper arm is essentially l3+l4 long
    x = np.cos(q1) * (l2 * np.cos(q2) + l3+l4 * np.cos(q2 + q3))
    y = np.sin(q1) * (l2 * np.cos(q2) + l3+l4 * np.cos(q2 + q3))
    z = l1 + l2 * np.sin(q2) + l3+l4 * np.sin(q2 + q3)
    end_coord = [x + x_b, y + y_b, z]
    return end_coord

def get_path_length(path):
    discretized_path = discretize_joint_space_path(path, 0.01)  # going in steps of 1% of the total movement per time
    path_length = 0
    joint_length = 0
    joint_travel = np.array([0., 0., 0., 0., 0.])
    x_b, y_b, q1, q2, q3 = discretized_path[0]
    coords = [FK(0.2, 1.0, 0.7, 0.2, x_b, y_b, q1, q2, q3)]
    joint_space = [discretized_path[0]]
    
    weights = [1, 1, 1, 1, 1]  # should modify this based on own constraints
    
    for i, orientation in enumerate(discretized_path):
        x_b, y_b, q1, q2, q3 = orientation
        coords.append(FK(0.2, 1.0, 0.7, 0.2, x_b, y_b, q1, q2, q3))
        joint_space.append(orientation)
        
        diff = np.array(coords[-1]) - np.array(coords[-2])  # difference in physical space
        diff2 = np.array(joint_space[-1]) - np.array(joint_space[-2]) # difference in joint space * weights
        length = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
        length2 = np.linalg.norm(diff2)  # lenght of joint space segment
        path_length += length
        joint_length += length2
        joint_travel += np.abs(diff2)
    return path_length, joint_length, joint_travel
    
'''test'''
'''
print(get_path_length([np.array([0.        , 0.        , 0.        , 0.        , 1.57079633]), np.array([-0.39796219,  0.89418043, -0.06444509,  0.36710845,  1.10890864]), np.array([-1.25922037,  1.63265571, -0.50440922,  0.3117727 ,  0.21041536]), np.array([-0.83007678,  2.79181365, -1.46086078,  0.82469706,  0.03499844]), np.array([-0.05094805,  4.01500958, -1.31670286,  1.24360605, -0.09504004]), np.array([ 0.97659763,  5.57664484, -1.19182613,  1.30966899,  0.08661594]), np.array([1.19878462, 6.87129503, 0.1373988 , 1.16390867, 0.20831629]), np.array([1.11235637, 7.43839145, 1.54707151, 0.8556802 , 0.48077969]), np.array([1.34421453, 7.53654684, 2.80957912, 0.67135383, 0.87816618])]))
'''
'''
print(get_path_length([np.array([0, 0, 0, 0, 0]),
		       np.array([1, 0, 0, 0, 0]),
		       np.array([2, 0, 0, 0, 0])]))
'''
