import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import do_mpc
import time
import math


# Define the model
model_type = 'continuous'
model = do_mpc.model.Model(model_type)

# Define state variables
x = model.set_variable(var_type='_x', var_name='x', shape=(2, 1))

# Define control input
u = model.set_variable(var_type='_u', var_name='u', shape=(2, 1))

# Define the dynamics: x_dot = u
model.set_rhs('x', u)

model.setup()

# Define the MPC controller
mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 30,
    't_step': 0.1,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 3,
    'collocation_ni': 2,
    'store_full_solution': True,
    'nlpsol_opts': {
        'ipopt.linear_solver': 'mumps',
          
    }
}

mpc.set_param(**setup_mpc)

# Define the cost function
#goal = np.array([-2.0, -1.0]).reshape(2, 1)
#goal = np.array([0.5, 1.0]).reshape(2, 1)
goal = np.array([1.0, -2.0]).reshape(2, 1)
mterm = (x - goal).T @ (x - goal)  # Terminal cost
lterm = (x - goal).T @ (x - goal) + u.T @ u  # Stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(u=0.1)  # Penalty on control input

mpc.bounds['lower', '_x', 'x'] = [-10, -10]
mpc.bounds['upper', '_x', 'x'] = [10, 10]
mpc.bounds['lower', '_u', 'u'] = [-1, -1]
mpc.bounds['upper', '_u', 'u'] = [1, 1]

# Define the square obstacle
box_min = np.array([0.0, 0.0])
box_max = np.array([1.0, 1.0])

# Define the constraint to avoid the square obstacle
def closest_point_on_box(px, py, box_min_x, box_min_y, box_max_x, box_max_y):
    """
    Find the closest point on a box to a given point (px, py) using CasADi.
    The box is defined by its minimum and maximum x and y coordinates.
    """
    closest_x = ca.fmax(ca.fmin(px, box_max_x), box_min_x)
    closest_y = ca.fmax(ca.fmin(py, box_max_y), box_min_y)

    return closest_x, closest_y

def distance_between_points(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points (x1, y1) and (x2, y2) using CasADi.
    """
    return ca.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def closest_point_and_distance(px, py, box_min_x, box_min_y, box_max_x, box_max_y):
    """
    Find the closest point on a box to a given point (px, py) and the distance between them using CasADi.
    The box is defined by its minimum and maximum x and y coordinates.
    """
    closest_x, closest_y = closest_point_on_box(px, py, box_min_x, box_min_y, box_max_x, box_max_y)
    distance = distance_between_points(px, py, closest_x, closest_y)

    return (closest_x, closest_y), distance

def J_x_t_func(gaussian):
    if len(gaussian) == 1:
        jxt = 0
    elif len(gaussian) > 1:
        jxt = sum([(gaussian[i] + gaussian[i - 1]) / 2 for i in range(1, len(gaussian))])
    else:
        jxt = 0

    return jxt

# Create CasADi SX variables
px, py = x[0], x[1]
box_min_x, box_min_y = ca.SX.sym('box_min_x'), ca.SX.sym('box_min_y')
box_max_x, box_max_y = ca.SX.sym('box_max_x'), ca.SX.sym('box_max_y')

closest_point, distance = closest_point_and_distance(px, py, box_min_x, box_min_y, box_max_x, box_max_y)

# Function to evaluate the closest point and distance
f = ca.Function('f', [px, py, box_min_x, box_min_y, box_max_x, box_max_y], [closest_point[0], closest_point[1], distance])

# Set the values for the obstacle box
box_min_values = [0.0, 0.0]
box_max_values = [1.0, 1.0]

# Evaluate the distance to the obstacle
obstacle_distance = f(x[0], x[1], *box_min_values, *box_max_values)[2]

# Add obstacle avoidance constraint
obstacle_constraint = -(obstacle_distance - 0.1)  # Add a small margin
mpc.set_nl_cons('obstacle_avoidance', obstacle_constraint, ub=0)

# Initialize variables for cumulative effect constraint
A = 0.5
#A = 1.0
sigma = 0.35
#sigma = 0.55
Jlimit = 0.2
dt = 0.1
quad_rad = 0.10

# Function to calculate the cumulative effect
def calculate_cumulative_effect(x_current, x_star, A, sigma, dt):
    norm = ca.norm_2(x_star - x_current)
    gaus = ca.if_else(norm <= ca.sqrt(A / sigma), A - sigma * norm**2, 0)
    return gaus

# Start timing
start_time = time.time()

# Create CasADi SX variable for cumulative effect
Jqt = ca.SX(0)
new = [Jqt]
#x_star_x, x_star_y = f(x[0], x[1], *box_min_values, *box_max_values)[0], f(x[0], x[1], *box_min_values, *box_max_values)[1]
#x_star = ca.vertcat(x_star_x, x_star_y)

(x_star_x, x_star_y), _ = closest_point_and_distance(x[0], x[1], *box_min_values, *box_max_values)

# Convert the tuple to a CasADi SX object
x_star = ca.vertcat(x_star_x, x_star_y)
# Accumulate the effect over the horizon
for i in range(setup_mpc['n_horizon']):
    Jqt += calculate_cumulative_effect(x, x_star, A, sigma, dt) * dt
    newn = calculate_cumulative_effect(x, x_star, A, sigma, dt)
    new.append(newn)

cumul = J_x_t_func(new)
cumul = cumul * dt
#print("JQT",Jqt)
#cumulative_constraint = -(Jlimit - Jqt)
cumulative_constraint = -(Jlimit - cumul)
mpc.set_nl_cons('cumulative_effect', cumulative_constraint, ub=0)

mpc.setup()

# Define the initial state
x0 = np.array([3.0, 3.0]).reshape(2, 1)
mpc.x0 = x0

# Set initial guess
mpc.set_initial_guess()

# Create the simulator
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=0.1)
simulator.setup()
simulator.x0 = x0

# Simulate and store the results
n_steps = 100
x_sim = np.zeros((n_steps + 1, 2))
x_sim[0, :] = x0.T



for k in range(n_steps):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    x_sim[k + 1, :] = x0.T



# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_sim[:, 0], x_sim[:, 1], 'o-', label='Trajectory')

plt.plot(goal[0], goal[1], 'rx', label='Goal')

# Add the square obstacle to the plot
square = plt.Rectangle(box_min, box_max[0] - box_min[0], box_max[1] - box_min[1], color='r', alpha=0.5, label='Obstacle')
plt.gca().add_patch(square)

# Add circles around each point in the trajectory
for point in x_sim:
    circle1 = plt.Circle((point[0] - quad_rad, point[1] + quad_rad), radius=quad_rad, fill=False, color='b')
    circle2 = plt.Circle((point[0] + quad_rad, point[1] + quad_rad), radius=quad_rad, fill=False, color='b')
    circle3 = plt.Circle((point[0] - quad_rad, point[1] - quad_rad), radius=quad_rad, fill=False, color='b')
    circle4 = plt.Circle((point[0] + quad_rad, point[1] - quad_rad), radius=quad_rad, fill=False, color='b')
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    plt.gca().add_patch(circle3)
    plt.gca().add_patch(circle4)

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Trajectory from Start to Goal with Square Obstacle Avoidance and Cumulative Effect Constraint')
plt.legend()
plt.grid()
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
#print (x_sim)

plt.show()




sigma = 0.35
A = 0.5
J_x_t_1 = []
parabolic_1 = []

sqrt_a_sigma = math.sqrt(A / sigma)
for t in range(1, len(x_sim) + 1):
    x_current = x_sim[t - 1]
    closest_point, distance = closest_point_and_distance(x_current[0], x_current[1], *box_min_values, *box_max_values)
    x_0_noise = np.array(closest_point).flatten()
    
    if np.linalg.norm(x_0_noise - x_current) <= sqrt_a_sigma:
        gaus = A - (sigma * (np.linalg.norm(x_0_noise - x_current) ** 2))
    else:
        gaus = 0
    parabolic_1.append(gaus)
    
    if len(parabolic_1) == 1:
        J_x_t = 0
    elif len(parabolic_1) > 1:
        J_x_of_t = J_x_t_func(parabolic_1)
        J_x_t = J_x_of_t * 0.1
        
    J_x_t_1.append(J_x_t)

# Print the values of J_x_t_1
#print("J_x_t_1 values:")
#print(J_x_t_1)

