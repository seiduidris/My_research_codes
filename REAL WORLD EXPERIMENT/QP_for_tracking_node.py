# #!/usr/bin/env python3
# #TO DEBUG
# from gurobipy import Model, GRB
# import numpy as np

# class QPTrajectoryTracker:
#     def __init__(self, config={}):
#         t1 = 1.0
#         t2 = 0.3
#         self.epsilon = 1.0
#         #self.epsilon = t1 * t2  # Finite time CLF coefficient
#         self.alpha_2 = t1 + t2  # Second constraint coefficient
#         self.gamma = 0.8  # Discount factor
#         self.J_limit = 7  # J limit
#         self.a = 0.4578  # Coefficient for ∇p^T
#         self.b = 0.1593  # Coefficient for ∇p^T
#         self.m = Model("CBF_CLF_QP")
#         self.num_of_states = 2
#         self.num_of_control_inputs = 2
#         self.u1_upper_lim = 10000
#         self.u1_lower_lim = -10000
#         self.u2_upper_lim = 10000
#         self.u2_lower_lim = -10000
#         self.dt = 0.1

#         # Control Variables
#         self.u1 = self.m.addVar(lb=self.u1_lower_lim, ub=self.u1_upper_lim, vtype=GRB.CONTINUOUS, name="x1_input_acceleration")
#         self.u2 = self.m.addVar(lb=self.u2_lower_lim, ub=self.u2_upper_lim, vtype=GRB.CONTINUOUS, name="x2_input_acceleration")
#         # Soft Constraint Variable for CLF
#         self.delta = self.m.addVar(lb=-30, ub=30, vtype=GRB.CONTINUOUS, name="relaxation_CLF")

#         # Initialize variables for J(q, t)
#         self.Jqt_value_mic1 = 0
#         self.previous_t_value = None

#     def J_x_t_func(self, gaussian):
#         """ Function to calculate the integral using the trapezoidal rule. """
#         gaussian = np.array(gaussian)
#         jxt = 0
#         if len(gaussian) > 1:
#             jxt = sum((gaussian[i] + gaussian[i - 1]) / 2 for i in range(1, len(gaussian)))
#         return jxt

#     def calculate_J(self, current_t, previous_t, db_history):
#         """ Calculate J(q,t) value based on sound data history. """
#         #segment_time_range = np.arange(previous_t, current_t, 1)[1:]
#         #segment_time_range = np.linspace(previous_t, current_t, num=max(2, int(current_t - previous_t) * 10))
#         segment_time_range = np.linspace(previous_t, current_t, num=max(10, round((current_t - previous_t) * 10)))

#         p_values = [self.get_db_at_time(t, db_history) for t in segment_time_range]
#         P_values = np.array([p for p in p_values])

#         # Apply gamma discount factor
#         gamma_factors = self.gamma ** (current_t - segment_time_range)
#         gaussian = gamma_factors * P_values
#         integral_approx = self.J_x_t_func(gaussian) * ((current_t - previous_t) / 10)
#         self.Jqt_value_mic1 = self.gamma ** (current_t - previous_t) * self.Jqt_value_mic1 + integral_approx
#         #print(f"segment_time_range: {segment_time_range}")
#         #print(f"p_values: {P_values}")
#         #print(f"db_history: {db_history}")
#         #print(f"current_t: {current_t}")
#         #print(f"previous_t: {previous_t}")
#         #print(f"gamma_factors: {gamma_factors}")
#         #print (f"integral_approx: {integral_approx}")
#        #print(f"Jqt_value_mic1 updated: {self.Jqt_value_mic1}, at time {current_t}")

#     def get_db_at_time(self, t, db_history):
#         """ Helper function to get the closest dB value from history for a given time t. """
#         closest_time = min(db_history, key=lambda x: abs(x[0] - t))
#         return closest_time[1]

#     def calculate_distance(self, x, q):
#         """ Calculate the Euclidean distance between two points. """
#         x = np.asarray(x).flatten()[:2]  # Ensure x is 2D (x, y)
#         q = np.asarray(q).flatten()[:2]  # Ensure q is 2D (x, y)
#         return np.linalg.norm(x - q)

#     def calculate_gradient_p_T(self, x, q):
#         """ Calculate the gradient ∇p^T(x, q) based on the given expression. """
#         d = self.calculate_distance(x, q)
#         x = np.asarray(x).flatten()[:2]
#         q = np.asarray(q).flatten()[:2]
#         grad_p_T = -2 * self.a * ((x - q).T / d) * (self.a * d + self.b) ** -3
#         return grad_p_T

#     def generate_control(self, x_current, x_ref, dx_ref, q_mic1, current_t, p_value, db_history, reference_velocities_times):
#         self.m.remove(self.m.getConstrs())

#         V = 0.5 * ((x_current[0] - x_ref[0])**2 + (x_current[1] - x_ref[1])**2)
#         grad_V_x1 = x_current[0] - x_ref[0]
#         grad_V_x2 = x_current[1] - x_ref[1]
      
        
#         dot_V_ref = -grad_V_x1 * dx_ref[0] - grad_V_x2 * dx_ref[1]
#         #self.m.addConstr(grad_V_x1 * self.u1 + grad_V_x2 * self.u2 + dot_V_ref + self.epsilon * V <= 0)
#         self.m.addConstr(grad_V_x1 * self.u1 + grad_V_x2 * self.u2 + dot_V_ref + self.epsilon * V <= 0)

#         if self.previous_t_value is not None:
#             self.calculate_J(current_t, self.previous_t_value, db_history)
#         self.previous_t_value = current_t


#         # Objective Function: Minimize \|u - \dot{x}_{\text{ref}}\|^2
#         #self.cost_func = (self.u1 - grad_V_x1)**2 + (self.u2 - grad_V_x2)**2
#         self.cost_func = (self.u1 - dx_ref[0])**2 + (self.u2 - dx_ref[1])**2
       
#         #self.cost_func = (self.u1 - dx_ref[0])**2 + (self.u2 - dx_ref[1])**2 
        
#         self.m.setObjective(self.cost_func, GRB.MINIMIZE)

#         # Optimize the QP
#         self.m.Params.LogToConsole = 0
#         self.m.optimize()

#         solution = self.m.getVars()
#         control_u1 = solution[0].x
#         control_u2 = solution[1].x

#         distance_to_goal_x = abs(x_current[0] - x_ref[0])
#         distance_to_goal_y = abs(x_current[1] - x_ref[1])

#         # scaled_u1 = control_u1 
#         # scaled_u2 = control_u2 

#         scaled_u1 = control_u1 * 35   #37    #47
#         scaled_u2 = control_u2 *  20      #20       #24

#         if distance_to_goal_x < 0.08 and abs(scaled_u1) < 5.0:
#             scaled_u1 = 5.0 * np.sign(control_u1)

#         if distance_to_goal_y < 0.08 and abs(scaled_u2) < 5.0:
#             scaled_u2 = 5.0 * np.sign(control_u2)

#         # New velocity and position update
#         target_vel = np.array([scaled_u1, scaled_u2])

#         target_pose = x_current + target_vel * self.dt
#         print(f"x_current: {x_current}")
#         print(f"x_ref: {x_ref}")
#         print(f"u_ref: {dx_ref}")
#         return target_vel, target_pose, self.Jqt_value_mic1






































#DRONE_TRACKING_RC
from gurobipy import Model, GRB
import numpy as np

class QPTrajectoryTracker:
    def __init__(self, config={}):
        t1 = 1.0
        t2 = 0.3
        self.epsilon = 1.0
        #self.epsilon = t1 * t2  # Finite time CLF coefficient
        self.alpha_2 = t1 + t2  # Second constraint coefficient
        self.gamma = 0.8  # Discount factor
        self.J_limit = 7  # J limit
        self.a = 0.4578  # Coefficient for ∇p^T
        self.b = 0.1593  # Coefficient for ∇p^T
        self.m = Model("CBF_CLF_QP")
        self.num_of_states = 2
        self.num_of_control_inputs = 2
        self.u1_upper_lim = 10000
        self.u1_lower_lim = -10000
        self.u2_upper_lim = 10000
        self.u2_lower_lim = -10000
        self.dt = 0.1

        # Control Variables
        self.u1 = self.m.addVar(lb=self.u1_lower_lim, ub=self.u1_upper_lim, vtype=GRB.CONTINUOUS, name="x1_input_acceleration")
        self.u2 = self.m.addVar(lb=self.u2_lower_lim, ub=self.u2_upper_lim, vtype=GRB.CONTINUOUS, name="x2_input_acceleration")
        # Soft Constraint Variable for CLF
        self.delta = self.m.addVar(lb=-30, ub=30, vtype=GRB.CONTINUOUS, name="relaxation_CLF")

        # Initialize variables for J(q, t)
        self.Jqt_value_mic1 = 0
        self.previous_t_value = None

    def J_x_t_func(self, gaussian):
        """ Function to calculate the integral using the trapezoidal rule. """
        gaussian = np.array(gaussian)
        jxt = 0
        if len(gaussian) > 1:
            jxt = sum((gaussian[i] + gaussian[i - 1]) / 2 for i in range(1, len(gaussian)))
        return jxt

    def calculate_J(self, current_t, previous_t, db_history):
        """ Calculate J(q,t) value based on sound data history. """
        #segment_time_range = np.arange(previous_t, current_t, 1)[1:]
        #segment_time_range = np.linspace(previous_t, current_t, num=max(2, int(current_t - previous_t) * 10))
        segment_time_range = np.linspace(previous_t, current_t, num=max(10, round((current_t - previous_t) * 10)))

        p_values = [self.get_db_at_time(t, db_history) for t in segment_time_range]
        P_values = np.array([p for p in p_values])

        # Apply gamma discount factor
        gamma_factors = self.gamma ** (current_t - segment_time_range)
        gaussian = gamma_factors * P_values
        integral_approx = self.J_x_t_func(gaussian) * ((current_t - previous_t) / 10)
        self.Jqt_value_mic1 = self.gamma ** (current_t - previous_t) * self.Jqt_value_mic1 + integral_approx
        #print(f"segment_time_range: {segment_time_range}")
        #print(f"p_values: {P_values}")
        #print(f"db_history: {db_history}")
        #print(f"current_t: {current_t}")
        #print(f"previous_t: {previous_t}")
        #print(f"gamma_factors: {gamma_factors}")
        #print (f"integral_approx: {integral_approx}")
       #print(f"Jqt_value_mic1 updated: {self.Jqt_value_mic1}, at time {current_t}")

    def get_db_at_time(self, t, db_history):
        """ Helper function to get the closest dB value from history for a given time t. """
        closest_time = min(db_history, key=lambda x: abs(x[0] - t))
        return closest_time[1]

    def calculate_distance(self, x, q):
        """ Calculate the Euclidean distance between two points. """
        x = np.asarray(x).flatten()[:2]  # Ensure x is 2D (x, y)
        q = np.asarray(q).flatten()[:2]  # Ensure q is 2D (x, y)
        return np.linalg.norm(x - q)

    def calculate_gradient_p_T(self, x, q):
        """ Calculate the gradient ∇p^T(x, q) based on the given expression. """
        d = self.calculate_distance(x, q)
        x = np.asarray(x).flatten()[:2]
        q = np.asarray(q).flatten()[:2]
        grad_p_T = -2 * self.a * ((x - q).T / d) * (self.a * d + self.b) ** -3
        return grad_p_T

    def generate_control(self, x_current, x_ref, dx_ref, q_mic1, current_t, p_value, db_history, reference_velocities_times):
        self.m.remove(self.m.getConstrs())

        V = 0.5 * ((x_current[0] - x_ref[0])**2 + (x_current[1] - x_ref[1])**2)
        grad_V_x1 = x_current[0] - x_ref[0]
        grad_V_x2 = x_current[1] - x_ref[1]

        
        dot_V_ref = -grad_V_x1 * dx_ref[0] - grad_V_x2 * dx_ref[1]
        #self.m.addConstr(grad_V_x1 * self.u1 + grad_V_x2 * self.u2 + dot_V_ref + self.epsilon * V <= 0)
        self.m.addConstr(grad_V_x1 * self.u1 + grad_V_x2 * self.u2 + dot_V_ref + self.epsilon * V <= 0)

        if self.previous_t_value is not None:
            self.calculate_J(current_t, self.previous_t_value, db_history)
        self.previous_t_value = current_t


        # Objective Function: Minimize \|u - \dot{x}_{\text{ref}}\|^2
        #self.cost_func = (self.u1 - grad_V_x1)**2 + (self.u2 - grad_V_x2)**2
        self.cost_func = (self.u1 - dx_ref[0])**2 + (self.u2 - dx_ref[1])**2
       
        #self.cost_func = (self.u1 - dx_ref[0])**2 + (self.u2 - dx_ref[1])**2 
        
        self.m.setObjective(self.cost_func, GRB.MINIMIZE)

        # Optimize the QP
        self.m.Params.LogToConsole = 0
        self.m.optimize()

        solution = self.m.getVars()
        control_u1 = solution[0].x
        control_u2 = solution[1].x

        distance_to_goal_x = abs(x_current[0] - x_ref[0])
        distance_to_goal_y = abs(x_current[1] - x_ref[1])

        # scaled_u1 = control_u1 
        # scaled_u2 = control_u2 

        scaled_u1 = control_u1 * 45   #37    #47
        scaled_u2 = control_u2 *  40      #20       #24

        if distance_to_goal_x > 0.08 and abs(scaled_u1) < 5.0:
            scaled_u1 = 5.0 * np.sign(control_u1)

        if distance_to_goal_y > 0.08 and abs(scaled_u2) < 5.0:
            scaled_u2 = 5.0 * np.sign(control_u2)

        # New velocity and position update
        target_vel = np.array([scaled_u1, scaled_u2])

        target_pose = x_current + target_vel * self.dt
        print(f"target_vel: {target_vel}")
        print(f"x_current: {x_current}")
        print(f"x_ref: {x_ref}")
        print(f"u_ref: {dx_ref}")
        return target_vel, target_pose, self.Jqt_value_mic1






























# #CODE FOR LIVE SOUND CONSTRAINT


# #!/usr/bin/env python3
# from gurobipy import Model, GRB
# import numpy as np

# class QPTrajectoryTracker:
#     def __init__(self, config={}):
#         t1 = 0.6
#         t2 = 0.3
        
#         self.epsilon = t1 * t2  # Finite time CLF coefficient
#         self.alpha_2 = t1 + t2  # Second constraint coefficient
#         self.gamma = 0.5  # Discount factor
#         self.J_limit = 40  # J limit
#         self.a = 0.4578  # Coefficient for ∇p^T
#         self.b = 0.1593  # Coefficient for ∇p^T
#         self.m = Model("CBF_CLF_QP")
#         self.num_of_states = 2
#         self.num_of_control_inputs = 2
#         self.u1_upper_lim = 10000
#         self.u1_lower_lim = -10000
#         self.u2_upper_lim = 10000
#         self.u2_lower_lim = -10000
#         self.dt = 0.1

#         # Control Variables
#         self.u1 = self.m.addVar(lb=self.u1_lower_lim, ub=self.u1_upper_lim, vtype=GRB.CONTINUOUS, name="x1_input_acceleration")
#         self.u2 = self.m.addVar(lb=self.u2_lower_lim, ub=self.u2_upper_lim, vtype=GRB.CONTINUOUS, name="x2_input_acceleration")
#         # Soft Constraint Variable for CLF
#         self.delta = self.m.addVar(lb=-30, ub=30, vtype=GRB.CONTINUOUS, name="relaxation_CLF")

#         # Initialize variables for J(q, t)
#         self.Jqt_value_mic1 = 0
#         self.previous_t_value = None

#     def J_x_t_func(self, gaussian):
#         """ Function to calculate the integral using the trapezoidal rule. """
#         gaussian = np.array(gaussian)
#         jxt = 0
#         if len(gaussian) > 1:
#             jxt = sum((gaussian[i] + gaussian[i - 1]) / 2 for i in range(1, len(gaussian)))
#         return jxt

#     def J_x_t_func(self, time_points, gaussian):
#         """ Function to calculate the integral using the trapezoidal rule for variable time steps. """
#         if len(gaussian) <= 1 or len(time_points) != len(gaussian):
#             return 0.0
        
#         total = 0.0
#         for i in range(1, len(gaussian)):
#             dt = time_points[i] - time_points[i - 1]  # Variable time step
#             total += ((gaussian[i] + gaussian[i - 1]) / 2.0) * dt
#         return total

#     def calculate_J(self, current_t, previous_t, db_history):
#         # Gather only the points in [previous_t, current_t]
#         time_db_pairs = [(t, db) for (t, db) in db_history if previous_t <= t <= current_t]
#         # Sort them by ascending time
#         time_db_pairs.sort(key=lambda x: x[0])
        
#         # Separate back into lists, now guaranteed ascending
#         time_points = [p[0] for p in time_db_pairs]
#         p_values    = [p[1] for p in time_db_pairs]
#         # print(f"[DEBUG] About to filter time_db_pairs. current_t={current_t}, previous_t={previous_t}")
#         # print(f"[DEBUG] time_db_pairs: {time_db_pairs}")
#         if len(time_points) < 2:
#             print("[DEBUG] Only 0 or 1 point found — returning early.")
#             return self.Jqt_value_mic1

#         # Apply discount factor
#         gamma_factors = self.gamma ** (current_t - np.array(time_points))
#         gaussian = gamma_factors * np.array(p_values)

#         # Compute the trapezoidal integral on these sorted points
#         integral_approx = self.J_x_t_func(time_points, gaussian)

#         self.Jqt_value_mic1 = (self.gamma ** (current_t - previous_t) * 
#                             self.Jqt_value_mic1 + 
#                             integral_approx)
#         # print(f"DB segment times (timepoints): {time_points}")
#         # print(f"Sorted? {time_points == sorted(time_points)}")
#         # print(f"Time deltas: {[time_points[i] - time_points[i-1] for i in range(1,len(time_points))]}")
#         # print(f"current_t: {current_t}")
#         # print(f"previous_t: {previous_t}")
#         # print(f"integral_approx: {integral_approx}")
#         # print(f"gamma_factor: {gamma_factors}")
#         #self.Jqt_value_mic1 = 20
#         return self.Jqt_value_mic1


#     # def calculate_J(self, current_t, previous_t, db_history):
#     #     """ Calculate J(q,t) value based on sound data history. """
        
#     #     # Extract segment_time_range directly from db_history
#     #     #time_points = sorted([t for t, _ in db_history if previous_t <= t <= current_t])
#     #     time_points = [t for t, _ in db_history if previous_t <= t <= current_t]

#     #     if len(time_points) < 2:
#     #         return self.Jqt_value_mic1  # Avoid division by zero if not enough data points

#     #     # Retrieve corresponding dB values
#     #     p_values = [db for t, db in db_history if t in time_points]

#     #     P_values = np.array(p_values)  # Convert to numpy array for calculations

#     #     # Apply gamma discount factor
#     #     gamma_factors = self.gamma ** (current_t - np.array(time_points))
#     #     gaussian = gamma_factors * P_values

#     #     # Correct integration using proper time steps
#     #     integral_approx = self.J_x_t_func(time_points, gaussian)

#     #     # Update J(q,t) value
#     #     self.Jqt_value_mic1 = self.gamma ** (current_t - previous_t) * self.Jqt_value_mic1 + integral_approx
#     #     print(f"DB segment times (timepoints): {time_points}")
#     #     print(f"Sorted? {time_points == sorted(time_points)}")
#     #     print(f"Time deltas: {[time_points[i] - time_points[i-1] for i in range(1,len(time_points))]}")
#     #     print(f"current_t: {current_t}")
#     #     print(f"previous_t: {previous_t}")

#     #     return self.Jqt_value_mic1

#     # def calculate_J(self, current_t, previous_t, db_history):
#     #     """ Calculate J(q,t) value based on sound data history. """
      
#     #     segment_time_range = [t for t, _ in db_history if previous_t <= t <= current_t]

#     #     if len(segment_time_range) < 2:
#     #         return self.Jqt_value_mic1  # Avoid division by zero if not enough data points

#     #     # Retrieve corresponding dB values
#     #     p_values = [db for t, db in db_history if t in segment_time_range]

#     #     P_values = np.array(p_values)  # Convert to numpy array for calculations

#     #     # Apply gamma discount factor
#     #     gamma_factors = self.gamma ** (current_t - np.array(segment_time_range))
#     #     gaussian = gamma_factors * P_values
#     #     integral_approx = self.J_x_t_func(gaussian) * ((current_t - previous_t) / len(segment_time_range))
#     #     self.Jqt_value_mic1 = self.gamma ** (current_t - previous_t) * self.Jqt_value_mic1 + integral_approx
#     #     #print(f"segment_time_range: {segment_time_range}")
#     #     #print(f"p_values: {P_values}")
#     #     #print(f"db_history: {db_history}")
#     #     #print(f"current_t: {current_t}")
#     #     #print(f"previous_t: {previous_t}")
#     #     #print(f"gamma_factors: {gamma_factors}")
#     #     #print (f"integral_approx: {integral_approx}")
#     #    #print(f"Jqt_value_mic1 updated: {self.Jqt_value_mic1}, at time {current_t}")
#     #     return self.Jqt_value_mic1

#     def get_db_at_time(self, t, db_history):
#         """ Helper function to get the closest dB value from history for a given time t. """
#         closest_time = min(db_history, key=lambda x: abs(x[0] - t))
#         return closest_time[1]

#     def calculate_distance(self, x, q):
#         """ Calculate the Euclidean distance between two points. """
#         x = np.asarray(x).flatten()[:2]  # Ensure x is 2D (x, y)
#         q = np.asarray(q).flatten()[:2]  # Ensure q is 2D (x, y)
#         return np.linalg.norm(x - q)

#     def calculate_gradient_p_T(self, x, q):
#         """ Calculate the gradient ∇p^T(x, q) based on the given expression. """
#         d = self.calculate_distance(x, q)
#         x = np.asarray(x).flatten()[:2]
#         q = np.asarray(q).flatten()[:2] 
#         grad_p_T = -2 * self.a * ((x - q) / d) * (self.a * d + self.b) ** -3
#         #return grad_p_T.reshape(1, -1)
#         return grad_p_T
    
#     def add_sound_constraint(self, x_current, q_mic1, jqt_value, p_value):
#         """Add the sound constraint to the model."""
#         grad_p_T = self.calculate_gradient_p_T(x_current, q_mic1)
#         log_gamma = np.log(self.gamma)
#         #first_term = -log_gamma * (0)
#         first_term = -log_gamma * (log_gamma * jqt_value + p_value)
#         second_term = -(grad_p_T[0] * self.u1 + grad_p_T[1] * self.u2)
#         third_term = self.epsilon * (-log_gamma * jqt_value - p_value)
#         fourth_term = self.alpha_2 * (-log_gamma * jqt_value - p_value +
#                                       self.epsilon *
#                                       (self.J_limit - jqt_value))
#         self.m.addConstr(
#             first_term + second_term + third_term + fourth_term >= 0,
#             "Second_constraint")

#     def generate_control(self, x_current, x_ref, dx_ref, q_mic1, current_t, p_value, db_history, reference_velocities_times):
#         print(f"[DEBUG] generate_control: previous_t_value={self.previous_t_value}, current_time={current_t}")
#         self.m.remove(self.m.getConstrs())

#         # Compute the distance function V(x) = 0.5 * d^2(x, x_ref)
#         x_error=x_ref - x_current
#         n_error=np.linalg.norm(x_error)
#         V=n_error
#         if n_error<1e-2:
#             grad_v=np.zeros((2,1))
#         else:
#             grad_v=x_error/n_error

#         grad_V_x1=grad_v[0].item()
#         grad_V_x2=grad_v[1].item()

#         # print (f"x_current: {x_current.tolist()}")
#         # print (f"x_ref: {x_ref.tolist()}")
#         # print (f"x_error: {x_error.tolist()}")
#         # print (f"grad_v: {grad_v.tolist()}")
        
#         #V = 0.5 * ((x_ref[0] - x_current[0])**2 + (x_ref[1] - x_current[1])**2)
#         #grad_V_x1 = (x_ref[0] - x_current[0])
#         #grad_V_x2 = (x_ref[1] - x_current[1])
                       
#         # Compute J(q, t) using the current time and previous time
#         if self.previous_t_value is not None:
#             jqt_value = self.calculate_J(current_t, self.previous_t_value, db_history)
#             #jqt_value = 20
#         else: 
#             jqt_value = 0
#         self.previous_t_value = current_t
#         grad_p_T = self.calculate_gradient_p_T(x_current, q_mic1)
        
#         self.add_sound_constraint(x_current, q_mic1, jqt_value, p_value)
#         # Objective Function: Minimize \|u - \alpha\dot{x}_{\text{ref}}\|^2
#         alpha=25 # Tune to match the expected speed of the robot
#         self.cost_func = (self.u1 - alpha*grad_V_x1)**2 + (self.u2 - alpha*grad_V_x2)**2
        
#         self.m.setObjective(self.cost_func, GRB.MINIMIZE)

#         # Optimize the QP
#         self.m.Params.LogToConsole = 0
#         self.m.optimize()

#         solution = self.m.getVars()
#         control_u1 = solution[0].x
#         control_u2 = solution[1].x

#         # if self.Jqt_value_mic1 > self.J_limit and abs(control_u1 * 32) < 5.0:
#         #     scaled_u1 = 5.0 * np.sign(control_u1)
#         # else:
#         distance_to_goal_x = abs(x_current[0] - x_ref[0])
#         distance_to_goal_y = abs(x_current[1] - x_ref[1])

#         # scaled_u1 = control_u1 * 15
#         # scaled_u2 = control_u2 
#         scaled_u1 = control_u1 
#         scaled_u2 = control_u2

#         # if distance_to_goal_x < 0.08 and abs(control_u1 * 1.2) < 5.0:
#         #     #if self.Jqt_value_mic1 < self.J_limit:
#         #     scaled_u1 = 5.0 * np.sign(control_u1)
#         # else:
#         #     scaled_u1 = control_u1 * 1.2

#         # if distance_to_goal_y < 0.08 and abs(control_u2 * 1.2) < 5.0:
#         #     #if self.Jqt_value_mic1 < self.J_limit:
#         #     scaled_u2 = 5.0 * np.sign(control_u2)
#         # else:
#         #     scaled_u2 = control_u2 * 1.2

#         if scaled_u2 >= 30:
#             scaled_u2 = 30
#         if scaled_u1 >= 30:
#             scaled_u1 = 30
#         #New velocity and position update
#         target_vel = np.array([scaled_u1, scaled_u2])
#         print (f"target_vel: {target_vel}")

#         target_pose = x_current + target_vel * self.dt
#         grad_v_flat = grad_v.flatten()
#         grad_p_T_flat = grad_p_T.flatten()
#         print (f"grad_v: {grad_v_flat}")
#         print (f"grad_sound: {grad_p_T_flat}")
#         print (f"q_mic1: {q_mic1}")
#         print (f"Jqt_value: {self.Jqt_value_mic1}")
#         print (f"Jqt_valueE: {jqt_value}")

#         return target_vel, x_current, self.Jqt_value_mic1, grad_v_flat, grad_p_T_flat








# #DRONE_TRACKING_RC_WITH_SOUND _CONSTRAINT



# #!/usr/bin/env python3
# from gurobipy import Model, GRB
# import numpy as np

# class QPTrajectoryTracker:
#     def __init__(self, config={}):
#         t1 = 0.6
#         t2 = 0.3
#         #self.epsilon = 1
#         self.epsilon = t1 * t2  # Finite time CLF coefficient
#         self.alpha_2 = t1 + t2  # Second constraint coefficient
#         self.gamma = 0.5  # Discount factor
#         self.J_limit = 40  # J limit
#         self.a = 0.4578  # Coefficient for ∇p^T
#         self.b = 0.1593  # Coefficient for ∇p^T
#         self.m = Model("CBF_CLF_QP")
#         self.num_of_states = 2
#         self.num_of_control_inputs = 2
#         self.u1_upper_lim = 10000
#         self.u1_lower_lim = -10000
#         self.u2_upper_lim = 10000
#         self.u2_lower_lim = -10000
#         self.dt = 0.1

#         # Control Variables
#         self.u1 = self.m.addVar(lb=self.u1_lower_lim, ub=self.u1_upper_lim, vtype=GRB.CONTINUOUS, name="x1_input_acceleration")
#         self.u2 = self.m.addVar(lb=self.u2_lower_lim, ub=self.u2_upper_lim, vtype=GRB.CONTINUOUS, name="x2_input_acceleration")
#         # Soft Constraint Variable for CLF
#         self.delta = self.m.addVar(lb=-30, ub=30, vtype=GRB.CONTINUOUS, name="relaxation_CLF")

#         # Initialize variables for J(q, t)
#         self.Jqt_value_mic1 = 0
#         self.previous_t_value = None

#     def J_x_t_func(self, gaussian):
#         """ Function to calculate the integral using the trapezoidal rule. """
#         gaussian = np.array(gaussian)
#         jxt = 0
#         if len(gaussian) > 1:
#             jxt = sum((gaussian[i] + gaussian[i - 1]) / 2 for i in range(1, len(gaussian)))
#         return jxt

#     def J_x_t_func(self, time_points, gaussian):
#         """ Function to calculate the integral using the trapezoidal rule for variable time steps. """
#         if len(gaussian) <= 1 or len(time_points) != len(gaussian):
#             return 0.0
        
#         total = 0.0
#         for i in range(1, len(gaussian)):
#             dt = time_points[i] - time_points[i - 1]  # Variable time step
#             total += ((gaussian[i] + gaussian[i - 1]) / 2.0) * dt
#         return total

#     def calculate_J(self, current_t, previous_t, db_history):
#         # Gather only the points in [previous_t, current_t]
#         time_db_pairs = [(t, db) for (t, db) in db_history if previous_t <= t <= current_t]
#         # Sort them by ascending time
#         time_db_pairs.sort(key=lambda x: x[0])
        
#         # Separate back into lists, now guaranteed ascending
#         time_points = [p[0] for p in time_db_pairs]
#         p_values    = [p[1] for p in time_db_pairs]
#         # print(f"[DEBUG] About to filter time_db_pairs. current_t={current_t}, previous_t={previous_t}")
#         # print(f"[DEBUG] time_db_pairs: {time_db_pairs}")
#         if len(time_points) < 2:
#             print("[DEBUG] Only 0 or 1 point found — returning early.")
#             return self.Jqt_value_mic1

#         # Apply discount factor
#         gamma_factors = self.gamma ** (current_t - np.array(time_points))
#         gaussian = gamma_factors * np.array(p_values)

#         # Compute the trapezoidal integral on these sorted points
#         integral_approx = self.J_x_t_func(time_points, gaussian)

#         self.Jqt_value_mic1 = (self.gamma ** (current_t - previous_t) * 
#                             self.Jqt_value_mic1 + 
#                             integral_approx)
 
#         return self.Jqt_value_mic1



#     def get_db_at_time(self, t, db_history):
#         """ Helper function to get the closest dB value from history for a given time t. """
#         closest_time = min(db_history, key=lambda x: abs(x[0] - t))
#         return closest_time[1]

#     def calculate_distance(self, x, q):
#         """ Calculate the Euclidean distance between two points. """
#         x = np.asarray(x).flatten()[:2]  # Ensure x is 2D (x, y)
#         q = np.asarray(q).flatten()[:2]  # Ensure q is 2D (x, y)
#         return np.linalg.norm(x - q)

#     def calculate_gradient_p_T(self, x, q):
#         """ Calculate the gradient ∇p^T(x, q) based on the given expression. """
#         d = self.calculate_distance(x, q)
#         x = np.asarray(x).flatten()[:2]
#         q = np.asarray(q).flatten()[:2] 
#         grad_p_T = -2 * self.a * ((x - q) / d) * (self.a * d + self.b) ** -3
#         #return grad_p_T.reshape(1, -1)
#         return grad_p_T
    
#     def add_sound_constraint(self, x_current, q_mic1, jqt_value, p_value):
#         """Add the sound constraint to the model."""
#         grad_p_T = self.calculate_gradient_p_T(x_current, q_mic1)
#         log_gamma = np.log(self.gamma)
#         #first_term = -log_gamma * (0)
#         first_term = -log_gamma * (log_gamma * jqt_value + p_value)
#         second_term = -(grad_p_T[0] * self.u1 + grad_p_T[1] * self.u2)
#         third_term = self.epsilon * (-log_gamma * jqt_value - p_value)
#         fourth_term = self.alpha_2 * (-log_gamma * jqt_value - p_value +
#                                       self.epsilon *
#                                       (self.J_limit - jqt_value))
#         self.m.addConstr(
#             first_term + second_term + third_term + fourth_term >= 0,
#             "Second_constraint")

#     def generate_control(self, x_current, x_ref, dx_ref, q_mic1, current_t, p_value, db_history, reference_velocities_times):
#         print(f"[DEBUG] generate_control: previous_t_value={self.previous_t_value}, current_time={current_t}")
#         self.m.remove(self.m.getConstrs())
        
#         # V = 0.5 * ((x_current[0] - x_ref[0])**2 + (x_current[1] - x_ref[1])**2)
#         # grad_V_x1 = x_current[0] - x_ref[0]
#         # grad_V_x2 = x_current[1] - x_ref[1]
#         # grad_v = np.array([[grad_V_x1], [grad_V_x2]])

#         V = 0.5 * (( x_ref[0] -  x_current[0])**2 + (x_ref[1]  - x_current[1])**2)
#         grad_V_x1 = x_ref[0] - x_current[0] 
#         grad_V_x2 = x_ref[1]  - x_current[1] 
#         grad_v = np.array([[grad_V_x1], [grad_V_x2]])
      
#         dot_V_ref = grad_V_x1 * dx_ref[0] + grad_V_x2 * dx_ref[1]
#         # dot_V_ref = -grad_V_x1 * dx_ref[0] - grad_V_x2 * dx_ref[1]
#         # self.m.addConstr(grad_V_x1 * self.u1 + grad_V_x2 * self.u2 + dot_V_ref + self.epsilon * V <= 0)
#         self.m.addConstr(-grad_V_x1 * self.u1 - grad_V_x2 * self.u2 + dot_V_ref + self.epsilon * V <= 0)


#         # # Compute the distance function V(x) = 0.5 * d^2(x, x_ref)
#         # x_error=x_ref - x_current
#         # #x_error= x_current - x_ref 
#         # n_error=np.linalg.norm(x_error)
#         # V=n_error
#         # if n_error<1e-2:
#         #     grad_v=np.zeros((2,1))
#         # else:
#         #     grad_v=x_error/n_error

#         # grad_V_x1=grad_v[0].item()
#         # grad_V_x2=grad_v[1].item()

                       
#         # Compute J(q, t) using the current time and previous time
#         if self.previous_t_value is not None:
#             jqt_value = self.calculate_J(current_t, self.previous_t_value, db_history)
           
#         else: 
#             jqt_value = 0
#         self.previous_t_value = current_t
#         grad_p_T = self.calculate_gradient_p_T(x_current, q_mic1)
        
#         self.add_sound_constraint(x_current, q_mic1, jqt_value, p_value)
#         # Objective Function: Minimize \|u - \alpha\dot{x}_{\text{ref}}\|^2
#         alpha=150 # Tune to match the expected speed of the robot
#         #self.cost_func = (self.u1 - alpha * dx_ref[0])**2 + (self.u2 - alpha * dx_ref[1])**2
#         self.cost_func = (self.u1 - dx_ref[0])**2 + (self.u2 - dx_ref[1])**2
        
#         self.m.setObjective(self.cost_func, GRB.MINIMIZE)

#         # Optimize the QP
#         self.m.Params.LogToConsole = 0
#         self.m.optimize()

#         solution = self.m.getVars()
#         control_u1 = solution[0].x
#         control_u2 = solution[1].x

     
#         distance_to_goal_x = abs(x_current[0] - x_ref[0])
#         distance_to_goal_y = abs(x_current[1] - x_ref[1])

      
#         scaled_u1 = control_u1 
#         scaled_u2 = control_u2


#         if scaled_u2 >= 30:
#             scaled_u2 = 30
#         if scaled_u1 >= 30:
#             scaled_u1 = 30
#         #New velocity and position update


#         scaled_u1 = control_u1 * 70   #60   #37    #47
#         scaled_u2 = control_u2 * 45 #20    #60      #20       #24

#         if distance_to_goal_x < 0.08 and abs(scaled_u1) < 5.0:
#             scaled_u1 = 5.0 * np.sign(control_u1)

#         if distance_to_goal_y < 0.08 and abs(scaled_u2) < 5.0:
#             scaled_u2 = 5.0 * np.sign(control_u2)




#         target_vel = np.array([scaled_u1, scaled_u2])
#         print (f"target_vel: {target_vel}")

#         target_pose = x_current + target_vel * self.dt
#         grad_v_flat = grad_v.flatten()
#         grad_p_T_flat = grad_p_T.flatten()
#         print (f"grad_v: {grad_v_flat}")
#         print (f"grad_sound: {grad_p_T_flat}")
#         print (f"q_mic1: {q_mic1}")
#         print (f"Jqt_value: {self.Jqt_value_mic1}")
#         print (f"Jqt_valueE: {jqt_value}")

#         return target_vel, x_current, self.Jqt_value_mic1, grad_v_flat, grad_p_T_flat






























# #!/usr/bin/env python3
# from gurobipy import Model, GRB
# import numpy as np

# class QPTrajectoryTracker:
#     def __init__(self, config={}):
#         t1 = 1.0
#         t2 = 0.3
        
#         self.epsilon = t1 * t2  # Finite time CLF coefficient
#         self.alpha_2 = t1 + t2  # Second constraint coefficient
#         self.gamma = 0.8  # Discount factor
#         self.J_limit = 7  # J limit
#         self.a = 0.4578  # Coefficient for ∇p^T
#         self.b = 0.1593  # Coefficient for ∇p^T
#         self.m = Model("CBF_CLF_QP")
#         self.num_of_states = 2
#         self.num_of_control_inputs = 2
#         self.u1_upper_lim = 10000
#         self.u1_lower_lim = -10000
#         self.u2_upper_lim = 10000
#         self.u2_lower_lim = -10000
#         self.dt = 0.1

#         # Control Variables
#         self.u1 = self.m.addVar(lb=self.u1_lower_lim, ub=self.u1_upper_lim, vtype=GRB.CONTINUOUS, name="x1_input_acceleration")
#         self.u2 = self.m.addVar(lb=self.u2_lower_lim, ub=self.u2_upper_lim, vtype=GRB.CONTINUOUS, name="x2_input_acceleration")
#         # Soft Constraint Variable for CLF
#         self.delta = self.m.addVar(lb=-30, ub=30, vtype=GRB.CONTINUOUS, name="relaxation_CLF")

#         # Initialize variables for J(q, t)
#         self.Jqt_value_mic1 = 0
#         self.previous_t_value = None

#     def J_x_t_func(self, gaussian):
#         """ Function to calculate the integral using the trapezoidal rule. """
#         gaussian = np.array(gaussian)
#         jxt = 0
#         if len(gaussian) > 1:
#             jxt = sum((gaussian[i] + gaussian[i - 1]) / 2 for i in range(1, len(gaussian)))
#         return jxt

#     def calculate_J(self, current_t, previous_t, db_history):
#         """ Calculate J(q,t) value based on sound data history. """
#         #segment_time_range = np.arange(previous_t, current_t, 1)[1:]
#         #segment_time_range = np.linspace(previous_t, current_t, num=max(2, int(current_t - previous_t) * 10))
#         segment_time_range = np.linspace(previous_t, current_t, num=max(10, round((current_t - previous_t) * 10)))

#         p_values = [self.get_db_at_time(t, db_history) for t in segment_time_range]
#         P_values = np.array([p for p in p_values])

#         # Apply gamma discount factor
#         gamma_factors = self.gamma ** (current_t - segment_time_range)
#         gaussian = gamma_factors * P_values
#         integral_approx = self.J_x_t_func(gaussian) * ((current_t - previous_t) / 10)
#         self.Jqt_value_mic1 = self.gamma ** (current_t - previous_t) * self.Jqt_value_mic1 + integral_approx
#         print(f"segment_time_range: {segment_time_range}")
#         print(f"p_values: {P_values}")
#         #print(f"db_history: {db_history}")
#         print(f"current_t: {current_t}")
#         print(f"previous_t: {previous_t}")
#         #print(f"gamma_factors: {gamma_factors}")
#         #print (f"integral_approx: {integral_approx}")
#        #print(f"Jqt_value_mic1 updated: {self.Jqt_value_mic1}, at time {current_t}")

#     def get_db_at_time(self, t, db_history):
#         """ Helper function to get the closest dB value from history for a given time t. """
#         closest_time = min(db_history, key=lambda x: abs(x[0] - t))
#         return closest_time[1]

#     def calculate_distance(self, x, q):
#         """ Calculate the Euclidean distance between two points. """
#         x = np.asarray(x).flatten()[:2]  # Ensure x is 2D (x, y)
#         q = np.asarray(q).flatten()[:2]  # Ensure q is 2D (x, y)
#         return np.linalg.norm(x - q)

#     def calculate_gradient_p_T(self, x, q):
#         """ Calculate the gradient ∇p^T(x, q) based on the given expression. """
#         d = self.calculate_distance(x, q)
#         x = np.asarray(x).flatten()[:2]
#         q = np.asarray(q).flatten()[:2]
#         grad_p_T = -2 * self.a * ((x - q).T / d) * (self.a * d + self.b) ** -3
#         return grad_p_T

#     def generate_control(self, x_current, x_ref, dx_ref, q_mic1, current_t, p_value, db_history, reference_velocities_times):
#         self.m.remove(self.m.getConstrs())

#         # Compute the distance function V(x) = 0.5 * d^2(x, x_ref)
#         V = 0.5 * ((x_current[0] - x_ref[0])**2 + (x_current[1] - x_ref[1])**2)
#         grad_V_x1 = (x_current[0] - x_ref[0])
#         grad_V_x2 = (x_current[1] - x_ref[1])

#         dot_V_ref = - grad_V_x1 * dx_ref[0] - grad_V_x2 * dx_ref [1]

#         # CLF Constraint: \dot{V} < cV
#         self.m.addConstr(grad_V_x1 * self.u1 + grad_V_x2 * self.u2 + dot_V_ref + self.epsilon * V <= 0, "Relaxed_CLF_constraint")
        
#         # Compute J(q, t) using the current time and previous time
#         if self.previous_t_value is not None:
#             self.calculate_J(current_t, self.previous_t_value, db_history)
#         self.previous_t_value = current_t

#         # Determine if the second constraint should activate
#         activate_second_constraint = any(abs(current_t - t) < 0.1 for t in reference_velocities_times)

#         # Add the second constraint only if within the specific times
#         if activate_second_constraint:
#             grad_p_T = self.calculate_gradient_p_T(x_current, q_mic1)
#             log_gamma = np.log(self.gamma)
#             first_term = -log_gamma * (log_gamma * self.Jqt_value_mic1 + p_value)
#             second_term = -(grad_p_T[0] * self.u1 + grad_p_T[1] * self.u2)
#             third_term = self.alpha_2 * (-log_gamma * self.Jqt_value_mic1 - p_value + self.epsilon * (self.J_limit - self.Jqt_value_mic1))
#             self.m.addConstr(first_term + second_term + third_term >= 0, "Second_constraint")

#         # Objective Function: Minimize \|u - \dot{x}_{\text{ref}}\|^2
#         self.cost_func = (self.u1 - dx_ref[0])**2 + (self.u2 - dx_ref[1])**2
#         self.m.setObjective(self.cost_func, GRB.MINIMIZE)

#         # Optimize the QP
#         self.m.Params.LogToConsole = 0
#         self.m.optimize()

#         solution = self.m.getVars()
#         control_u1 = solution[0].x
#         control_u2 = solution[1].x

#         # Determine if second constraint should activate and apply appropriate scaling
#         if activate_second_constraint:
#             # Scaling logic when second constraint is active
#             if self.Jqt_value_mic1 > self.J_limit and abs(control_u1 * 32) < 5.0:
#                 scaled_u1 = 5.0 * np.sign(control_u1)
#             else:
#                 distance_to_goal_x = abs(x_current[0] - x_ref[0])
#                 distance_to_goal_y = abs(x_current[1] - x_ref[1])

#                 scaled_u1 = control_u1 * 32
#                 scaled_u2 = control_u2 * 32

#                 if distance_to_goal_x > 0.08 and abs(scaled_u1) < 5.0:
#                     if self.Jqt_value_mic1 < self.J_limit:
#                         scaled_u1 = 5.0 * np.sign(control_u1)

#                 if distance_to_goal_y > 0.08 and abs(scaled_u2) < 5.0:
#                     if self.Jqt_value_mic1 < self.J_limit:
#                         scaled_u2 = 5.0 * np.sign(control_u2)

#         else:
#             # Scaling logic when second constraint is not activee
#             distance_to_goal_x = abs(x_current[0] - x_ref[0])
#             distance_to_goal_y = abs(x_current[1] - x_ref[1])

#             scaled_u1 = control_u1 * 32
#             scaled_u2 = control_u2 * 32

#             if distance_to_goal_x > 0.08 and abs(scaled_u1) < 5.0:
#                 scaled_u1 = 5.0 * np.sign(control_u1)

#             if distance_to_goal_y > 0.08 and abs(scaled_u2) < 5.0:
#                 scaled_u2 = 5.0 * np.sign(control_u2)

#         # New velocity and position update
#         target_vel = np.array([scaled_u1, scaled_u2])

#         target_pose = x_current + target_vel * self.dt

#         return target_vel, target_pose, self.Jqt_value_mic1






