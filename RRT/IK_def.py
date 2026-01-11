import numpy as np

def IK_endpoint(l1, l2, l3, l4, exclusion_radius, end_point, upper=True):
    	'''
    	Performs simple calculation to get end configuration of '2-DOF' arm (on rotating pedestal)
    	input: 
    		(l1-l4): joint lengths, four for our robot
    		- l1: base height
    		- l2: link 1 length
    		- l3: link 2 length
    		- l4: end effector block length, middle is taken as endpoint
    		end_point: endpoint coordinates
    		upper: True or False, default True, takes the upper configuration or else lower
    	ouput:
    		q_end: endpoint configuration (x,y,z)
    		or
    		error message: out of reach
    	'''    	
    	
    	#origin = base_end+[l1]  # coordinates base
    	#diff = np.array(end_point) - np.array(origin)
    	
    	# check if endpoint is within reach (assuming base is at 0,0,l1)
    	#distance = np.linalg.norm(diff)
    	#if l2-(l3+l4/2) > distance > l2+(l3+l4/2):
    	#    return f'Point {end_point} is out of reach!'
    	
    	# Get random valid base endpoint spot
    	height_diff = end_point[2] - l1  # endpoint z - base (arm joint) z
    	max_distance_radius = np.sqrt((l2+l3+l4/2)**2 - height_diff**2)  # Pythagoras, a**2 + b**2 = c**2

    	# sample random coordinates for the base until they satisfy x**2 + y**2 <= r**2
    	while True:
    	   x_b = np.random.uniform(-max_distance_radius, max_distance_radius)  # relative x to endpoint x 
    	   y_b = np.random.uniform(-max_distance_radius, max_distance_radius)  # relative y to endpoint y
    	   if exclusion_radius**2 <= x_b**2 + y_b**2 <= max_distance_radius**2:
    	      break
    	
    	origin = [end_point[0] + x_b, end_point[1] + y_b, l1]  # random coordinates base
    	diff = np.array(end_point) - np.array(origin)
    	distance = np.linalg.norm(diff)
    	
    	# if valid, get spherical coordinates:
    	x,y,z = diff
    	r = distance
    	phi = np.arctan2(y, x)
    	theta = np.arctan2(z , np.sqrt(x**2 + y**2))
    	q1 = phi
    	# law of cosines:
    	a = l3+l4/2
    	b = r
    	c = l2
    	#print(f'theta: {theta}')
    	q2 = np.arccos( (a**2 - b**2 - c**2) / (-2*b*c) )
    	q3 = np.pi - np.arccos( (b**2 - a**2 - c**2) / (-2*a*c) )
    	
    	if upper:
    		q2 = np.pi/2 - theta - q2
    	else:
    		q2 = np.pi/2 - theta + q2
    		q3 = -q3
    	
    	#q_end = np.array([base_end[0], base_end[1], q1, q2, q3])
    	q_end = np.array([origin[0], origin[1], q1, q2, q3])
    	return q_end

def IK_endpoint2(l1, l2, l3, l4, base_q, exclusion_radius, base_coord, end_point, upper=True):
    	'''
    	Performs simple calculation to get end configuration of '2-DOF' arm (on rotating pedestal)
    	input: 
    		(l1-l4): joint lengths, four for our robot
    		- l1: base height
    		- l2: link 1 length
    		- l3: link 2 length
    		- l4: end effector block length, middle is taken as endpoint
    		end_point: endpoint coordinates
    		upper: True or False, default True, takes the upper configuration or else lower
    	ouput:
    		q_end: endpoint configuration (x,y,z)
    		or
    		error message: out of reach
    	'''    	
    	
    	'''
    	#check if current position is within reach of radius goal
    	diff = np.array(end_point) - np.array(base_coord)
    	
    	# check if endpoint is within reach (assuming base is at 0,0,l1)
    	distance = np.linalg.norm(diff)
    	if l2-(l3+l4/2) > distance > l2+(l3+l4/2):
    	#    return f'Point {end_point} is out of reach!'
    	    return None
    	'''
    	
    	# Get maximum radius around endpoint in 2D
    	height_diff = end_point[2] - l1  # endpoint z - base (arm joint) z
    	max_distance_radius = np.sqrt((l2+l3+l4/2)**2 - height_diff**2)  # Pythagoras, a**2 + b**2 = c**2
    	
    	# our base position is given
    	x_b = base_coord[0]
    	y_b = base_coord[1]
    	
    	point_radius = (x_b-end_point[0])**2 + (y_b-end_point[1])**2
    	# check if the base position is inside the radius to the endpoint
    	if point_radius > max_distance_radius**2 or point_radius < exclusion_radius**2:
    		#print('Base too far from the endpoint')
    		return None
    	
    	origin = [x_b, y_b, l1]  # random coordinates base
    	diff = np.array(end_point) - np.array(origin)
    	distance = np.linalg.norm(diff)
    	
    	# if valid, get spherical coordinates:
    	x,y,z = diff
    	r = distance
    	phi = np.arctan2(y, x)
    	theta = np.arctan2(z , np.sqrt(x**2 + y**2))
    	q1 = phi
    	# law of cosines:
    	a = l3+l4/2
    	b = r
    	c = l2
    	#print(f'theta: {theta}')
    	q2 = np.arccos( (a**2 - b**2 - c**2) / (-2*b*c) )
    	q3 = np.pi - np.arccos( (b**2 - a**2 - c**2) / (-2*a*c) )
    	
    	if upper:
    		q2 = np.pi/2 - theta - q2
    	else:
    		q2 = np.pi/2 - theta + q2
    		q3 = -q3
    	
    	#q_end = np.array([base_end[0], base_end[1], q1, q2, q3])
    	# modify q1 to fit to current robot orientation
    	# this circumenvents the need to account for wrapping around the base rotation (which is fully S1)
    	# whilst sampling, or calculating joint space lenghts
    	q1 = q1 + (2*np.pi) * round( (base_q-q1)/(2*np.pi) )
	
    	q_end = np.array([origin[0], origin[1], q1, q2, q3])
    	return q_end
