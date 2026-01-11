from pinocchio.visualize import MeshcatVisualizer
import time
import random
import pinocchio  # for custom urdf
import numpy as np

# for obstacles
import coal
from pyroboplan.core.utils import set_collisions

from pyroboplan.core.utils import get_random_collision_free_state
from pyroboplan.models.two_dof import load_models, add_object_collisions
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from rrt_mod import RRTPlanner2, RRTPlannerOptions2
from get_path_length import get_path_length
from IK_def import IK_endpoint
from pyroboplan.planning.utils import discretize_joint_space_path


if __name__ == "__main__":
    # Create models and data
    # Custom model TODO:
    # Change filename if needed to the right path to the robot urdf
    filename = "mobileManipulator.urdf"
    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(filename)
    
    
    #Show joint limits
    print('joint limits (x, y, q1, q2, q3):')
    print(model.lowerPositionLimit)
    print(model.upperPositionLimit)
    
    
    # LOADING ENVIRONMENT USING THE LOADER
    # import environment_loader_pin
    from environment_loader_pin import load_environment_from_txt
    
    # Loads obstacles, returning a list of obstacle names 
    # and adding their collisions to collision_model and visual_model
    
    ###################################################################################
    #				    INSERT SCENARIO				      #
    ###################################################################################
    
    scenario = 'scenario_6.5_obstacles.txt'
    q_start = np.array([-9, -9, 0, 0, 0]) # zero position: x, y, q1, q2, q3
    
    ###################################################################################
    #										      #
    ###################################################################################
    
    obstacle_names = load_environment_from_txt(scenario, visual_model, collision_model)
    
    # All obstacles:
    #print(obstacle_names)
    
    # Define the active collision pairs between the robot and obstacle links.
    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if not "obstacle" in cobj.name
    ]
    
    # All collision objects:
    #print([cobj.name for cobj in collision_model.geometryObjects])
    
    #obstacle_names = ["obstacle_1", "obstacle_2", "obstacle_3"]  # EDIT if more obstacles added
    for obstacle_name in [obname for obname in obstacle_names if obname != "goal"]:
        for collision_name in collision_names:
            set_collisions(model, collision_model, obstacle_name, collision_name, True)
            
    
    data = model.createData()
    collision_data = collision_model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    
    end_point = [0, 8, 1.0]   # goal
    exclusion_radius = 0.3  # m, radius from goal 2D plane coordinate to not use, mostly collision avoidance
    q_end = IK_endpoint(0.2, 1.0, 0.7, 0.2, exclusion_radius, end_point, upper=True)
    print(f'START: {q_start}')
    print(f'END: {q_end}')
    
    

    # Configure the RRT planner
    #seed = int(random.uniform(1, 9999999))
    
    options = RRTPlannerOptions(
        max_step_size=0.05,		# step size for checking collision -> smaller: more accurate (rad)
        max_connection_dist=1.0,	# maximum distance for connecting nodes (m or rad)
        rrt_connect=False,		# RRT connect extends samples from the most recent node
        bidirectional_rrt=False,	# make RRT path from both start->end and end->start
        rrt_star=True,			# RRT* rewires nodes within a radius to shortcut nodes
        max_rewire_dist=1.0,		# only used when using rrt*, however gets modified in the loop to theory
        max_planning_time=300.0,	# max time allowed planning RRT
        rng_seed=None,
        fast_return=True,		# stops planning when first valid solution is found
        goal_biasing_probability=0.1,	# probability of sampling the goal configuration
        collision_distance_padding=0.0,	# distance margin to obstacles
    )

    planner = RRTPlanner2(model, collision_model, options=options)

    while True:
        viz.display(q_start)
        time.sleep(0.5)

        # Search for a path
        #path = planner.plan(q_start, q_end)
        path = planner.plan(q_start, q_end, 0.2, 1.0, 0.7, 0.2, exclusion_radius, end_point)
        planner.visualize(viz, "tool_link", show_tree=True)
        #planner.visualize(viz, "base_main_link", show_tree=True) # for 2D base frame path

        # Animate the path
        if path:
            print(f'Room scenario:         	{scenario}')
            #print(f'Using random seed:     {seed}')
            print(f'path:                  	{path}')
            print(f'cartesian path length: 	{get_path_length(path)[0]}')
            print(f'joint path length:     	{get_path_length(path)[1]}')
            print(f'joint travel distances	{get_path_length(path)[2]}')
            discretized_path = discretize_joint_space_path(path, options.max_step_size)

            input("Press 'Enter' to animate the path.")
            
            for q in discretized_path:
                viz.display(q)  # uncomment/comment out if you want to see animation of path
                time.sleep(0.05)
            

            input("Press 'Enter' to plan another path, or ctrl-c to quit.")
            print()
            q_start = q_start #q_end
            q_end = IK_endpoint(0.2, 1.0, 0.7, 0.2, exclusion_radius, end_point, upper=True) #get_random_collision_free_state(model, collision_model)
            print(f'New endpoint configuration')
            print(f'END: {q_end}')
