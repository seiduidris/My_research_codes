import numpy as np
import matplotlib.pyplot as plt
import math
import random

random.seed(40)


# ------------------------------
# NODE CLASS
# ------------------------------
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0  # Euclidean cost
        self.time = 0.0  # Accumulated travel time
        self.pxq_values = None  # List of pxq values for each obstacle segment
        self.jxt_cost = None  # List of Jₓₜ cost values (one per segment)
        self.phi = None  # List of φ values (one per segment)


# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def distance(n1, n2):
    return math.hypot(n1.x - n2.x, n1.y - n2.y)


def point_to_segment_distance(point,
                              seg_start,
                              seg_end,
                              return_closest_point=False):
    """
    Computes the minimum distance from a point to a line segment.
    If return_closest_point=True, also returns (closest_x, closest_y).
    """
    px, py = point
    ax, ay = seg_start
    bx, by = seg_end
    apx = px - ax
    apy = py - ay
    abx = bx - ax
    aby = by - ay
    ab_squared = abx**2 + aby**2
    if ab_squared == 0:
        dist_val = math.hypot(apx, apy)
        if return_closest_point:
            return dist_val, (ax, ay)
        else:
            return dist_val
    t = (apx * abx + apy * aby) / ab_squared
    t = max(0, min(1, t))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    dist_val = math.hypot(px - closest_x, py - closest_y)
    if return_closest_point:
        return dist_val, (closest_x, closest_y)
    else:
        return dist_val


def get_obstacle_segments(obstacles):
    """
    Given obstacles as (xmin, ymin, xmax, ymax),
    returns a list of segments (each as ((x1,y1),(x2,y2))).
    Each obstacle contributes four segments.
    """
    segments = []
    for obs in obstacles:
        xmin, ymin, xmax, ymax = obs
        segments.append(((xmin, ymin), (xmax, ymin)))
        segments.append(((xmax, ymin), (xmax, ymax)))
        segments.append(((xmax, ymax), (xmin, ymax)))
        segments.append(((xmin, ymax), (xmin, ymin)))
    return segments


def compute_pxq_values(node, segments):
    """
    For a given node and list of segments, computes a list of pxq values.
    pxq = (a*d + b)^(-2) + c + 60,  with d=dist(node,segment).
    """
    a = 0.4578
    b = 0.1593
    c = -41.00
    pxq_values = []
    point = (node.x, node.y)
    for seg in segments:
        seg_start, seg_end = seg
        d_seg = point_to_segment_distance(point, seg_start, seg_end)
        val = a * d_seg + b
        if abs(val) < 1e-9:  # avoid division by zero
            val = 1e-6
        pxq = val**(-2) + c + 60
        pxq_values.append(pxq)
    return pxq_values


def J_x_t_funct(gaussian, dt=0.1):
    """
    Approximates the integral using the trapezoidal rule.
    'gaussian' is a list of function values at discretized time points.
    """
    if len(gaussian) <= 1:
        return 0.0
    else:
        total = 0.0
        for i in range(1, len(gaussian)):
            total += ((gaussian[i] + gaussian[i - 1]) / 2.0) * dt
        return total


def calc_new_jxt_cost(from_node, to_node, gamma=0.9, dt=0.1, segments=None):
    """
    Computes the new jxt_cost for to_node given its parent (from_node).
    Formula:
      J(q,t₂) = γ^(t₂-t₁)*J(q,t₁) + ∫ₜ₁ᵗ₂ γ^(t₂−τ)* p(q,x(τ)) dτ.
    """
    deltaT = to_node.time - from_node.time
    deltaX = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
    if from_node.jxt_cost is None:
        parent_jxt = [0.0 for _ in range(len(segments))]
    else:
        parent_jxt = from_node.jxt_cost

    # First term: gamma^(deltaT)*parent_jxt
    first_term = [(gamma**deltaT) * cost for cost in parent_jxt]

    # If there's no time difference, return just the scaled parent cost
    if abs(deltaT) < 1e-9:
        return first_term

    # Prepare for integral
    n_steps = max(1, int(deltaT / dt))
    tau_values = [from_node.time + i * dt for i in range(n_steps)]
    if tau_values[-1] < to_node.time:
        tau_values.append(to_node.time)

    t1 = from_node.time
    t2 = to_node.time
    f_values_per_seg = [[] for _ in range(len(segments))]
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y

    # Sample points along the line from from_node -> to_node
    for tau in tau_values:
        if deltaT < 1e-9:
            x_tau = from_node.x
            y_tau = from_node.y
        else:
            alpha = (tau - t1) / (t2 - t1)  # param in [0,1]
            x_tau = from_node.x + alpha * dx
            y_tau = from_node.y + alpha * dy

        temp_node = Node(x_tau, y_tau)
        temp_node.pxq_values = compute_pxq_values(temp_node, segments)
        weight = gamma**(t2 - tau)
        for j in range(len(segments)):
            f_values_per_seg[j].append(temp_node.pxq_values[j] * weight)

    integral_term = []
    for j in range(len(segments)):
        integ = J_x_t_funct(f_values_per_seg[j], dt)
        integral_term.append(integ)

    new_jxt = [first_term[j] + integral_term[j] for j in range(len(segments))]
    return new_jxt


def steer(from_node, to_point, step_size, segments):
    """
    Steers from from_node toward to_point by at most step_size.
    Also sets the new node's cost, time, pxq_values.
    """
    speed = 0.1
    dx = to_point[0] - from_node.x
    dy = to_point[1] - from_node.y
    dist = math.hypot(dx, dy)

    if dist <= step_size:
        new_x = to_point[0]
        new_y = to_point[1]
    else:
        theta = math.atan2(dy, dx)
        new_x = from_node.x + step_size * math.cos(theta)
        new_y = from_node.y + step_size * math.sin(theta)

    new_node = Node(new_x, new_y)
    new_node.parent = from_node
    d = math.hypot(new_x - from_node.x, new_y - from_node.y)
    new_node.cost = from_node.cost + d
    travel_time = d / speed
    new_node.time = from_node.time + travel_time

    # Compute pxq values for the newly created node
    new_node.pxq_values = compute_pxq_values(new_node, segments)
    return new_node


def is_inside_box(point, box):
    """
    Checks if point (x,y) is inside the given box (xmin, ymin, xmax, ymax).
    """
    x, y = point
    return (box[0] <= x <= box[2]) and (box[1] <= y <= box[3])


def collision_check(node1, node2, obstacles):
    """
    Checks if the straight-line path between node1 and node2 is collision-free
    by sampling points at a given resolution.
    """
    resolution = 0.01
    dist = math.hypot(node2.x - node1.x, node2.y - node1.y)
    steps = int(dist / resolution)
    if steps == 0:
        steps = 1
    for i in range(steps + 1):
        t = i / steps
        x = node1.x + t * (node2.x - node1.x)
        y = node1.y + t * (node2.y - node1.y)
        for obs in obstacles:
            if is_inside_box((x, y), obs):
                return False
    return True


def get_nearest_node(nodes, point):
    """
    Returns the node in 'nodes' that is nearest to the given 'point'.
    """
    min_dist = float('inf')
    nearest = nodes[0]
    for node in nodes:
        d = math.hypot(node.x - point[0], node.y - point[1])
        if d < min_dist:
            min_dist = d
            nearest = node
    return nearest


# ------------------------------
# PHI FUNCTION
# ------------------------------
def compute_phi_values(from_node,
                       to_node,
                       segments,
                       j_limit,
                       gamma=0.9,
                       alpha1=0.9,
                       alpha2=0.9,
                       a=0.4578,
                       b=0.1593):
    """
    Computes φ_{J,2} for each obstacle segment.

      φ_j = 
        -log(γ)*( log(γ)*J_j + p_j )
        + 2*a * (a*d + b)^(-3) * [((q - x)^T / d) * u]
        + α1[ -log(γ)*J_j - p_j ]
        + α2[ -log(γ)*J_j - p_j + α1( J_limit - J_j ) ]

    Returns one φ per segment.
    """
    phi_list = []
    delta_t = to_node.time - from_node.time
    # Velocity vector u
    if abs(delta_t) < 1e-9:
        u = np.array([0.0, 0.0])
    else:
        u = np.array([to_node.x - from_node.x, to_node.y - from_node.y
                      ]) / delta_t

    for j, seg in enumerate(segments):
        seg_start, seg_end = seg
        dist_j, (qx, qy) = point_to_segment_distance((to_node.x, to_node.y),
                                                     seg_start,
                                                     seg_end,
                                                     return_closest_point=True)
        # J and p
        if to_node.jxt_cost is None:
            J_j = 0.0
        else:
            J_j = to_node.jxt_cost[j]
        if to_node.pxq_values is None:
            p_j = 0.0
        else:
            p_j = to_node.pxq_values[j]

        # (q - x)
        q_minus_x = np.array([qx - to_node.x, qy - to_node.y])
        # dot product: ((q - x)^T / d_j) * u
        dot_val = 0.0
        if dist_j > 1e-9:
            dot_val = np.dot(q_minus_x, u) / dist_j

        # second_term = 2 * a * (a*dist_j + b)^(-3) * dot_val
        second_term = 2.0 * a * ((a * dist_j + b)**-3) * dot_val

        part1 = -math.log(gamma) * (math.log(gamma) * J_j + p_j)
        part3 = alpha1 * (-math.log(gamma) * J_j - p_j)
        part4 = alpha2 * (-math.log(gamma) * J_j - p_j + alpha1 *
                          (j_limit - J_j))

        phi_j = part1 + second_term + part3 + part4
        phi_list.append(phi_j)

    return phi_list


# ------------------------------
# GET PATH TO GOAL (with final step to actual goal)
# ------------------------------
def get_path_to_goal(nodes,
                     goal,
                     threshold=1.0,
                     segments=None,
                     gamma=0.9,
                     dt=0.1,
                     j_limit=7.5,
                     obstacles=None):
    """
    Extracts a path from the start to a node near the goal (within threshold).
    If that node is not exactly the goal, we attempt to connect to the goal
    (collision check + constraints). If successful, we append the exact goal node.

    Returns a list of Node objects if successful, otherwise None.
    """
    if not nodes:
        return None

    # Find all nodes within 'threshold' distance of the goal
    goal_candidates = []
    for node in nodes:
        if math.hypot(node.x - goal.x, node.y - goal.y) <= threshold:
            goal_candidates.append(node)

    if not goal_candidates:
        return None

    # Among candidates, pick the one with the smallest 'cost'
    best_node = min(goal_candidates, key=lambda n: n.cost)

    # Backtrack to get the path
    path = []
    current = best_node
    while current is not None:
        path.append(current)
        current = current.parent
    path.reverse()

    # If the final node is not exactly the goal, try to connect to the goal
    last = path[-1]
    if (abs(last.x - goal.x) > 1e-9) or (abs(last.y - goal.y) > 1e-9):
        # Check collision from 'last' to 'goal'
        if obstacles is not None:
            # Make a Node for the goal
            goal_node = Node(goal.x, goal.y)
            dist_to_goal = math.hypot(goal.x - last.x, goal.y - last.y)
            speed = 0.1
            goal_node.time = last.time + (dist_to_goal / speed)
            goal_node.parent = last
            goal_node.pxq_values = compute_pxq_values(goal_node, segments)

            if collision_check(last, goal_node, obstacles):
                # Compute JXT for that final link
                goal_node_jxt = calc_new_jxt_cost(last, goal_node, gamma, dt,
                                                  segments)

                # Check j_limit
                if all(jc <= j_limit for jc in goal_node_jxt):
                    # Check φ ≥ 0
                    goal_node.jxt_cost = goal_node_jxt
                    goal_node_phi = compute_phi_values(last,
                                                       goal_node,
                                                       segments,
                                                       j_limit,
                                                       gamma=gamma,
                                                       alpha1=0.9,
                                                       alpha2=0.9)
                    if all(pval >= 0 for pval in goal_node_phi):
                        goal_node.phi = goal_node_phi
                        path.append(goal_node)  # Attach the exact goal
                    else:
                        print("  -> Final node REJECTED: some φ < 0.")
                else:
                    print(
                        "  -> Final node REJECTED: jxt_cost exceeds j_limit.")
            else:
                print("  -> Final node REJECTED: collision with obstacle.")

    return path


# ------------------------------
# RRT* ALGORITHM
# ------------------------------
def rrt_star(start,
             goal,
             obstacles,
             search_area,
             max_iter=500,
             step_size=1,
             goal_sample_rate=0.2,
             neighbor_radius=2,
             threshold=1.00,
             gamma=0.9,
             dt=0.1,
             j_limit=7.5):
    """
    Runs the RRT* algorithm with a JXT cost constraint + φ≥0 constraint.
    Returns:
      - nodes: all nodes in the tree.
      - path_progress: list of (best_path_length, #nodes) whenever a new best is found.
    """
    segments = get_obstacle_segments(obstacles)

    # Initialize the start node
    start.pxq_values = compute_pxq_values(start, segments)
    start.jxt_cost = [0.0 for _ in range(len(segments))]
    start.phi = [0.0 for _ in range(len(segments))]  # trivial at start
    nodes = [start]

    best_path_length = float('inf')
    path_progress = []
    speed = 0.1

    for i in range(max_iter):
        # Sample random point or goal
        if random.random() < goal_sample_rate:
            sample = (goal.x, goal.y)
        else:
            sample = (random.uniform(search_area[0], search_area[1]),
                      random.uniform(search_area[2], search_area[3]))

        # Get nearest node and steer
        nearest_node = get_nearest_node(nodes, sample)
        new_node = steer(nearest_node, sample, step_size, segments)

        # Collision check from nearest_node to new_node
        if not collision_check(nearest_node, new_node, obstacles):
            continue  # skip if collision

        # Find near nodes (for re-parenting)
        near_nodes = []
        for node in nodes:
            if distance(node, new_node) <= neighbor_radius:
                if collision_check(node, new_node, obstacles):
                    near_nodes.append(node)

        # Ensure the nearest_node is in near_nodes
        if nearest_node not in near_nodes:
            near_nodes.append(nearest_node)

        valid_parents = []
        for candidate in near_nodes:
            d_candidate = distance(candidate, new_node)
            travel_time = d_candidate / speed

            # Temporarily set 'to_node' times
            tmp_node = Node(new_node.x, new_node.y)
            tmp_node.time = candidate.time + travel_time
            tmp_cost = calc_new_jxt_cost(candidate, tmp_node, gamma, dt,
                                         segments)

            # Check JXT limit
            if all(jc <= j_limit for jc in tmp_cost):
                # Check φ ≥ 0
                tmp_node.jxt_cost = tmp_cost
                tmp_node.pxq_values = new_node.pxq_values
                tmp_phi = compute_phi_values(candidate,
                                             tmp_node,
                                             segments,
                                             j_limit,
                                             gamma=gamma,
                                             alpha1=0.9,
                                             alpha2=0.9)
                if all(pval >= 0 for pval in tmp_phi):
                    tentative_cost = candidate.cost + d_candidate
                    valid_parents.append(
                        (candidate, tentative_cost, tmp_cost, tmp_phi))
                # else: φ < 0 => invalid
            # else: JXT over limit => invalid

        if not valid_parents:
            # No valid parent that satisfies the constraints
            continue

        # Among valid parents, pick the one with the lowest path cost
        valid_parents.sort(key=lambda x: x[1])
        best_parent, best_cost, best_jxt, best_phi = valid_parents[0]

        # Update new_node with that best parent
        new_node.parent = best_parent
        new_node.cost = best_cost
        d_parent = distance(best_parent, new_node)
        new_node.time = best_parent.time + (d_parent / speed)
        new_node.jxt_cost = best_jxt
        new_node.phi = best_phi

        # Add new_node to the tree
        nodes.append(new_node)

        # Rewire near_nodes
        for near_node in near_nodes:
            if near_node is new_node:
                continue
            d_rewire = distance(new_node, near_node)
            potential_cost = new_node.cost + d_rewire
            if potential_cost < near_node.cost:
                # Check JXT constraints from new_node to near_node
                tmp_rewire_node = Node(near_node.x, near_node.y)
                tmp_rewire_node.time = new_node.time + (d_rewire / speed)
                tmp_rewire_jxt = calc_new_jxt_cost(new_node, tmp_rewire_node,
                                                   gamma, dt, segments)
                if all(jc <= j_limit for jc in tmp_rewire_jxt):
                    tmp_rewire_node.jxt_cost = tmp_rewire_jxt
                    tmp_rewire_node.pxq_values = near_node.pxq_values
                    tmp_phi = compute_phi_values(new_node,
                                                 tmp_rewire_node,
                                                 segments,
                                                 j_limit,
                                                 gamma=gamma,
                                                 alpha1=0.9,
                                                 alpha2=0.9)
                    if all(pval >= 0 for pval in tmp_phi):
                        # Rewire
                        near_node.parent = new_node
                        near_node.cost = potential_cost
                        near_node.time = tmp_rewire_node.time
                        near_node.jxt_cost = tmp_rewire_jxt
                        near_node.phi = tmp_phi

        # Try to see if we have a path to the goal now
        current_path = get_path_to_goal(nodes,
                                        goal,
                                        threshold=threshold,
                                        segments=segments,
                                        gamma=gamma,
                                        dt=dt,
                                        j_limit=j_limit,
                                        obstacles=obstacles)
        if current_path is not None:
            # Compute path length
            coords = [(n.x, n.y) for n in current_path]
            current_length = compute_path_length(coords)
            if current_length < best_path_length:
                best_path_length = current_length
                path_progress.append((best_path_length, len(nodes)))

        print(f"Iter: {i}, number of nodes: {len(nodes)}")

    return nodes, path_progress


def compute_path_length(path_coords):
    """
    Given a list of (x,y) tuples, computes the total path length.
    """
    length = 0.0
    for i in range(1, len(path_coords)):
        length += math.hypot(path_coords[i][0] - path_coords[i - 1][0],
                             path_coords[i][1] - path_coords[i - 1][1])
    return length


def calculate_total_jxt_for_path(path, segments, gamma=0.9, dt=0.1):
    """
    Given a path (list of Node objects), re-compute the
    cumulative jxt cost from start to end by chaining calls to calc_new_jxt_cost.
    """
    if len(path) < 2:
        return None

    # Start from path[0] -> path[1]
    cumulative_jxt = calc_new_jxt_cost(path[0], path[1], gamma, dt, segments)
    for i in range(2, len(path)):
        dummy_parent = Node(path[i - 1].x, path[i - 1].y)
        dummy_parent.time = path[i - 1].time
        dummy_parent.jxt_cost = cumulative_jxt
        cumulative_jxt = calc_new_jxt_cost(dummy_parent, path[i], gamma, dt,
                                           segments)

    return cumulative_jxt


def plot_rrt(nodes, path, obstacles, start, goal, search_area):
    plt.figure(figsize=(7, 5))
    # Draw edges
    for node in nodes:
        if node.parent is not None:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")

    # Final path in red
    if path is not None and len(path) > 1:
        px = [p.x for p in path]
        py = [p.y for p in path]
        plt.plot(px, py, "-r", linewidth=2, label="Final path")

    # Obstacles
    for obs in obstacles:
        rect = plt.Rectangle((obs[0], obs[1]),
                             obs[2] - obs[0],
                             obs[3] - obs[1],
                             edgecolor="black",
                             fill=False,
                             linewidth=1.5)
        plt.gca().add_patch(rect)

    # Start and goal markers
    plt.plot(start.x, start.y, "bs", markersize=10, label="Start")
    plt.plot(goal.x, goal.y, "ms", markersize=10, label="Goal")

    plt.xlim(search_area[0], search_area[1])
    plt.ylim(search_area[2], search_area[3])
    plt.title("RRT* Path Planning with JXT & φ≥0 Constraints")
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.legend()
    plt.grid(True)
    plt.show()


# ------------------------------
# MAIN (Example)
# ------------------------------
def main():
    # Example search area, start, goal, obstacles
    search_area = (-5, 30, -5, 20)
    # search_area = (-10, 40, -10, 30)
    start = Node(5.5, 1.0)
    goal = Node(22.5, 10)

    obstacles = [
        (7.4, 2.3, 9.3, 3.8),
        (10.1, 2.3, 10.9, 4.5),
        (11.8, 2.5, 13.9, 4.9),
        (14.5, 3.7, 17.0, 4.9),
        (14.6, 2.1, 17.1, 3.3),
        (17.8, 4.0, 20.1, 5.2),
        (20.7, 2.3, 23.3, 3.6),
        (23.5, 0.0, 26.1, 1.9),
        (21.0, 6.2, 21.7, 7.1),
        (22.4, 6.2, 23.1, 7.1),
        (24.9, 7.6, 25.8, 9.2),
        (24.9, 9.6, 26.0, 11.2),
        (24.9, 13.6, 25.8, 14.5),
        (19.9, 11.8, 21.6, 14.1),
        (14.7, 11.4, 15.8, 12.8),
        (9.6, 10.8, 10.2, 12.3),
        (7.1, 10.8, 7.7, 12.3),
        (0.0, 6.1, 2.2, 12.2),
        (15.6, 6.2, 17.0, 9.8),
        (17.6, 8.3, 20.8, 11.6),
        (17.6, 6.3, 19.7, 8.1),
        (11.9, 6.0, 14.0, 8.3),
        (2.2, 8.4, 5.1, 10.3),
        (5.1, 6.4, 7.8, 9.9),
        (7.8, 8.0, 10.8, 9.7),
    ]

    max_iter = 6000
    step_size = 1
    #goal_sample_rate = 0.2
    goal_sample_rate = 0.2
    #neighbor_radius = 2
    neighbor_radius = 2.0
    threshold = 0.3
    j_limit = 197

    # Run RRT*
    nodes, path_progress = rrt_star(start,
                                    goal,
                                    obstacles,
                                    search_area,
                                    max_iter=max_iter,
                                    step_size=step_size,
                                    goal_sample_rate=goal_sample_rate,
                                    neighbor_radius=neighbor_radius,
                                    threshold=threshold,
                                    gamma=0.9,
                                    dt=0.1,
                                    j_limit=j_limit)

    # Extract final path
    # (this will also attempt the exact goal connection if feasible)
    final_path = get_path_to_goal(nodes,
                                  goal,
                                  threshold=threshold,
                                  segments=get_obstacle_segments(obstacles),
                                  gamma=0.9,
                                  dt=0.1,
                                  j_limit=j_limit,
                                  obstacles=obstacles)

    print("\nPath Length Samples (best path found over iterations):")
    print(path_progress)

    if final_path is not None and len(final_path) > 0:
        # Double-check JXT limit
        all_within_limit = True
        for node in final_path:
            if node.jxt_cost and any(jxt > j_limit for jxt in node.jxt_cost):
                all_within_limit = False
                break

        if all_within_limit:
            print("\nAll nodes in the final path satisfy j_limit.")
        else:
            print("\nWARNING: Some node(s) in the final path exceed j_limit!")

        print("\nFinal Path (Node, Time, pxq, jxt_cost, phi):")
        for node in final_path:
            print(f"Node: ({node.x:.2f}, {node.y:.2f}), Time: {node.time:.4f}")
            if node.pxq_values is not None:
                print("  pxq:", [f"{val:.4f}" for val in node.pxq_values])
            if node.jxt_cost is not None:
                print("  jxt:", [f"{val:.4f}" for val in node.jxt_cost])
            if node.phi is not None:
                print("  phi:", [f"{pval:.4f}" for pval in node.phi])

        # Compute total jxt over the path (should match final node's jxt if done consistently)
        total_jxt = calculate_total_jxt_for_path(
            final_path, get_obstacle_segments(obstacles), gamma=0.9, dt=0.1)
        print("\nTotal jxt cost for the final path:")
        if total_jxt:
            print([f"{val:.4f}" for val in total_jxt])
        else:
            print("No path or single-node path (nothing to compute).")
    else:
        print(
            "\nNo path found that satisfies j_limit and φ≥0 all the way to the goal!"
        )

    # Visualize
    plot_rrt(nodes, final_path, obstacles, start, goal, search_area)

    if final_path is not None and len(final_path) > 0:
        # Each box (obstacle) has 4 consecutive segments.
        # The first obstacle is at indices [0, 1, 2, 3].
        seg_indices = [0, 1, 2, 3]  # first obstacle's 4 edges
        j_limit = 197
        # Gather times and the corresponding jxt costs for those 4 segments
        times = [node.time for node in final_path]
        jxt_seg_0 = [node.jxt_cost[0] for node in final_path]  # segment 0
        jxt_seg_1 = [node.jxt_cost[1] for node in final_path]  # segment 1
        jxt_seg_2 = [node.jxt_cost[2] for node in final_path]  # segment 2
        jxt_seg_3 = [node.jxt_cost[3] for node in final_path]  # segment 3

        # Now plot the four lines on one figure
        plt.figure()
        plt.plot(times, jxt_seg_0, label='Segment 0')
        plt.plot(times, jxt_seg_1, label='Segment 1')
        plt.plot(times, jxt_seg_2, label='Segment 2')
        plt.plot(times, jxt_seg_3, label='Segment 3')
        plt.axhline(y=j_limit, color='r', linestyle='--', label='j_limit')

        plt.xlabel('Time')
        plt.ylabel('JXT Cost')
        plt.title('JXT vs. Time for First Obstacle (Segments 0..3)')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()

# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import random

# random.seed(40)

# # ------------------------------
# # NODE CLASS
# # ------------------------------
# class Node:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         self.parent = None
#         self.cost = 0.0  # Euclidean cost
#         self.time = 0.0  # Accumulated travel time
#         self.pxq_values = None  # List of pxq values for each obstacle segment (computed once)
#         self.jxt_cost = None  # List of Jₓₜ cost values (one per segment)
#         self.phi = None  # List of φ values (one per segment) -- ADDED

# # ------------------------------
# # HELPER FUNCTIONS
# # ------------------------------
# def distance(n1, n2):
#     return math.hypot(n1.x - n2.x, n1.y - n2.y)

# def point_to_segment_distance(point,
#                               seg_start,
#                               seg_end,
#                               return_closest_point=False):
#     """
#     Computes the minimum distance from a point to a line segment.
#     If return_closest_point=True, also returns (closest_x, closest_y).
#     """
#     px, py = point
#     ax, ay = seg_start
#     bx, by = seg_end
#     apx = px - ax
#     apy = py - ay
#     abx = bx - ax
#     aby = by - ay
#     ab_squared = abx**2 + aby**2
#     if ab_squared == 0:
#         dist_val = math.hypot(apx, apy)
#         if return_closest_point:
#             return dist_val, (ax, ay)
#         else:
#             return dist_val
#     t = (apx * abx + apy * aby) / ab_squared
#     t = max(0, min(1, t))
#     closest_x = ax + t * abx
#     closest_y = ay + t * aby
#     dist_val = math.hypot(px - closest_x, py - closest_y)
#     if return_closest_point:
#         return dist_val, (closest_x, closest_y)
#     else:
#         return dist_val

# def get_obstacle_segments(obstacles):
#     """
#     Given obstacles as (xmin, ymin, xmax, ymax),
#     returns a list of segments (each as ((x1,y1),(x2,y2))).
#     Each obstacle contributes four segments.
#     """
#     segments = []
#     for obs in obstacles:
#         xmin, ymin, xmax, ymax = obs
#         segments.append(((xmin, ymin), (xmax, ymin)))
#         segments.append(((xmax, ymin), (xmax, ymax)))
#         segments.append(((xmax, ymax), (xmin, ymax)))
#         segments.append(((xmin, ymax), (xmin, ymin)))
#     return segments

# def compute_pxq_values(node, segments):
#     """
#     For a given node and list of segments, computes a list of pxq values.
#     For each segment, using the formula:
#       pxq = (a * d + b)^(-2) + c + 60,
#     where d is the distance from the node to the segment,
#           a = 0.4578, b = 0.1593, c = -41.00.
#     """
#     a = 0.4578
#     b = 0.1593
#     c = -41.00
#     pxq_values = []
#     point = (node.x, node.y)
#     for seg in segments:
#         seg_start, seg_end = seg
#         d_seg = point_to_segment_distance(point, seg_start, seg_end)
#         val = a * d_seg + b
#         if abs(val) < 1e-9:  # avoid division by zero
#             val = 1e-6
#         pxq = val**(-2) + c + 60
#         pxq_values.append(pxq)
#     return pxq_values

# def J_x_t_funct(gaussian, dt=0.1):
#     """
#     Approximates the integral using the trapezoidal rule.
#     'gaussian' is a list of function values at discretized time points.
#     """
#     if len(gaussian) <= 1:
#         return 0.0
#     else:
#         total = 0.0
#         for i in range(1, len(gaussian)):
#             total += ((gaussian[i] + gaussian[i - 1]) / 2.0) * dt
#         return total

# def calc_new_jxt_cost(from_node, to_node, gamma=0.9, dt=0.1, segments=None):
#     """
#     Computes the new jxt_cost for to_node given its parent (from_node)
#     using the recursive formula:
#       J(q,t₂) = γ^(t₂-t₁) * J(q,t₁) + ∫ₜ₁ᵗ₂ γ^(t₂ - τ) p(q, x(τ)) dτ.
#     This is done for each obstacle segment.

#     We discretize the path in space or time.
#     (Below uses distance-based steps to create tau_values, for instance.)
#     """
#     deltaT = to_node.time - from_node.time
#     deltaX = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
#     if from_node.jxt_cost is None:
#         parent_jxt = [0.0 for _ in range(len(segments))]
#     else:
#         parent_jxt = from_node.jxt_cost

#     # First term: gamma^(deltaT)*parent_jxt
#     first_term = [(gamma**deltaT) * cost for cost in parent_jxt]

#     # If there's no time difference, return the scaled parent cost
#     if abs(deltaT) < 1e-9:
#         return first_term

#     # Prepare for integral
#     n_steps = max(1, int(deltaX / dt))
#     tau_values = [from_node.time + i * dt for i in range(n_steps)]
#     if tau_values[-1] < to_node.time:
#         tau_values.append(to_node.time)

#     t1 = from_node.time
#     t2 = to_node.time
#     f_values_per_seg = [[] for _ in range(len(segments))]
#     dx = to_node.x - from_node.x
#     dy = to_node.y - from_node.y

#     for tau in tau_values:
#         alpha = (tau - t1) / deltaX if deltaX > 1e-9 else 0
#         x_tau = from_node.x + alpha * dx
#         y_tau = from_node.y + alpha * dy
#         temp_node = Node(x_tau, y_tau)
#         temp_node.pxq_values = compute_pxq_values(temp_node, segments)
#         weight = gamma**(t2 - tau)
#         for j in range(len(segments)):
#             f_values_per_seg[j].append(temp_node.pxq_values[j] * weight)

#     integral_term = []
#     for j in range(len(segments)):
#         integ = J_x_t_funct(f_values_per_seg[j], dt)
#         integral_term.append(integ)

#     new_jxt = [first_term[j] + integral_term[j] for j in range(len(segments))]
#     return new_jxt

# def steer(from_node, to_point, step_size, segments):
#     """
#     Steers from from_node toward to_point by at most step_size.
#     Computes the new node’s cost, time, and its pxq_values.
#     """
#     speed = 0.1
#     dx = to_point[0] - from_node.x
#     dy = to_point[1] - from_node.y
#     dist = math.hypot(dx, dy)

#     if dist <= step_size:
#         new_x = to_point[0]
#         new_y = to_point[1]
#     else:
#         theta = math.atan2(dy, dx)
#         new_x = from_node.x + step_size * math.cos(theta)
#         new_y = from_node.y + step_size * math.sin(theta)

#     new_node = Node(new_x, new_y)
#     new_node.parent = from_node
#     d = math.hypot(new_x - from_node.x, new_y - from_node.y)
#     new_node.cost = from_node.cost + d
#     travel_time = d / speed
#     new_node.time = from_node.time + travel_time

#     # Compute pxq values for the newly created node
#     new_node.pxq_values = compute_pxq_values(new_node, segments)
#     return new_node

# def is_inside_box(point, box):
#     """
#     Checks if point (x,y) is inside the given box (xmin, ymin, xmax, ymax).
#     """
#     x, y = point
#     return (box[0] <= x <= box[2]) and (box[1] <= y <= box[3])

# def collision_check(node1, node2, obstacles):
#     """
#     Checks if the straight-line path between node1 and node2 is collision-free.
#     """
#     resolution = 0.1
#     dist = math.hypot(node2.x - node1.x, node2.y - node1.y)
#     steps = int(dist / resolution)
#     if steps == 0:
#         steps = 1
#     for i in range(steps + 1):
#         t = i / steps
#         x = node1.x + t * (node2.x - node1.x)
#         y = node1.y + t * (node2.y - node1.y)
#         for obs in obstacles:
#             if is_inside_box((x, y), obs):
#                 return False
#     return True

# def get_nearest_node(nodes, point):
#     """
#     Returns the node in 'nodes' that is nearest to the given 'point'.
#     """
#     min_dist = float('inf')
#     nearest = nodes[0]
#     for node in nodes:
#         d = math.hypot(node.x - point[0], node.y - point[1])
#         if d < min_dist:
#             min_dist = d
#             nearest = node
#     return nearest

# # ------------------------------
# # PHI FUNCTION
# # ------------------------------
# def compute_phi_values(from_node,
#                        to_node,
#                        segments,
#                        j_limit,
#                        gamma=0.9,
#                        alpha1=9,
#                        alpha2=9,
#                        a=0.4578,
#                        b=0.1593):
#     """
#     Computes φ_{J,2} for each obstacle segment:

#       φ = -log(γ)*[log(γ)*J + p]
#           + [2 * a * ((q - x)^T / d(q,x)) * (a*d(q,x) + b)^(-3)] * u
#           + α1[-log(γ)*J - p]
#           + α2[-log(γ)*J - p + α1(J_limit - J)]

#     Returns a list (one φ per segment).

#     Notes:
#       - J = to_node.jxt_cost[j]
#       - p = to_node.pxq_values[j]
#       - d(q,x) is the distance from the node's position to the segment j.
#       - (q - x) is the vector from node.x,y to that closest point q on the segment.
#       - u = ( (to_node.x - from_node.x), (to_node.y - from_node.y) ) / (to_node.time - from_node.time).
#         If delta_t=0, we skip or set u=0.
#     """
#     phi_list = []
#     delta_t = to_node.time - from_node.time
#     # Construct velocity vector u
#     if abs(delta_t) < 1e-9:
#         u = np.array([0.0, 0.0])
#     else:
#         u = np.array([to_node.x - from_node.x, to_node.y - from_node.y
#                       ]) / delta_t

#     # For each segment j, compute the corresponding φ
#     for j, seg in enumerate(segments):
#         seg_start, seg_end = seg
#         # Distance and the closest point
#         dist_j, (qx, qy) = point_to_segment_distance((to_node.x, to_node.y),
#                                                      seg_start,
#                                                      seg_end,
#                                                      return_closest_point=True)
#         # J and p
#         if to_node.jxt_cost is None:
#             # fallback if something is not set
#             J_j = 0.0
#         else:
#             J_j = to_node.jxt_cost[j]
#         if to_node.pxq_values is None:
#             p_j = 0.0
#         else:
#             p_j = to_node.pxq_values[j]

#         # (q - x)
#         q_minus_x = np.array([qx - to_node.x, qy - to_node.y])
#         # dot product: ((q - x)^T / d_j) * u
#         dot_val = 0.0
#         if dist_j > 1e-9:
#             dot_val = np.dot(q_minus_x, u) / dist_j

#         # The second big bracket factor:
#         second_term = 2.0 * a * ((a * dist_j + b)**-3) * dot_val

#         # Build the phi expression
#         # -log(gamma)*(log(gamma)*J_j + p_j)
#         part1 = -math.log(gamma) * (math.log(gamma) * J_j + p_j)
#         # α1 * [-log(gamma)*J_j - p_j]
#         part3 = alpha1 * (-math.log(gamma) * J_j - p_j)
#         # α2 * [ -log(gamma)*J_j - p_j + α1*(J_limit - J_j) ]
#         part4 = alpha2 * (-math.log(gamma) * J_j - p_j + alpha1 *
#                           (j_limit - J_j))

#         phi_j = part1 + second_term + part3 + part4

#         phi_list.append(phi_j)

#     return phi_list

# # ------------------------------
# # RRT* ALGORITHM
# # ------------------------------
# def rrt_star(start,
#              goal,
#              obstacles,
#              search_area,
#              max_iter=500,
#              step_size=1,
#              goal_sample_rate=0.2,
#              neighbor_radius=2,
#              threshold=1.0,
#              gamma=0.9,
#              dt=0.1,
#              j_limit=15):
#     """
#     Runs the RRT* algorithm with a JXT cost constraint + φ≥0 constraint.
#     Returns:
#       - nodes: all nodes in the tree.
#       - path_progress: list of (best_path_length, #nodes_at_that_time).
#     """
#     segments = get_obstacle_segments(obstacles)

#     # Initialize the start node
#     start.pxq_values = compute_pxq_values(start, segments)
#     start.jxt_cost = [0.0 for _ in range(len(segments))]
#     start.phi = [0.0 for _ in range(len(segments))]  # no movement => trivial

#     nodes = [start]
#     best_path_length = float('inf')
#     path_progress = []
#     speed = 0.1

#     for i in range(max_iter):
#         # Sample random point or goal
#         if random.random() < goal_sample_rate:
#             sample = (goal.x, goal.y)
#         else:
#             sample = (random.uniform(search_area[0], search_area[1]),
#                       random.uniform(search_area[2], search_area[3]))

#         # Get nearest node and steer
#         nearest_node = get_nearest_node(nodes, sample)
#         new_node = steer(nearest_node, sample, step_size, segments)

#         # Collision check
#         if not collision_check(nearest_node, new_node, obstacles):
#             # Skipping due to collision
#             continue

#         # Find near nodes for potential re-parenting
#         near_nodes = []
#         for node in nodes:
#             if distance(node, new_node) <= neighbor_radius:
#                 if collision_check(node, new_node, obstacles):
#                     near_nodes.append(node)

#         # Try to find a valid parent among near_nodes (plus the nearest_node itself)
#         if nearest_node not in near_nodes:
#             near_nodes.append(nearest_node)

#         valid_parents = []
#         for candidate in near_nodes:
#             # Compute the new jxt cost if 'new_node' is connected to 'candidate'
#             d_candidate = distance(candidate, new_node)
#             travel_time = d_candidate / speed

#             # Temporarily set 'to_node' times for the cost calculation
#             tmp_node = Node(new_node.x, new_node.y)
#             tmp_node.time = candidate.time + travel_time
#             tmp_cost = calc_new_jxt_cost(candidate, tmp_node, gamma, dt,
#                                          segments)

#             # Check JXT cost limit
#             if all(jc <= j_limit for jc in tmp_cost):
#                 # Now check φ >= 0
#                 tmp_node.jxt_cost = tmp_cost
#                 tmp_node.pxq_values = new_node.pxq_values  # same position => same pxq
#                 tmp_phi = compute_phi_values(candidate,
#                                              tmp_node,
#                                              segments,
#                                              j_limit,
#                                              gamma=gamma,
#                                              alpha1=9,
#                                              alpha2=9)
#                 # print(
#                 #     f"\nChecking φ for potential parent => Candidate({candidate.x:.2f},{candidate.y:.2f}),"
#                 #     f" NewNode({tmp_node.x:.2f},{tmp_node.y:.2f}):\n  φ-values = {[f'{pval:.4f}' for pval in tmp_phi]}"
#                 # )

#                 if all(pval >= 0 for pval in tmp_phi):
#                     # If valid, record the parent, the cost, jxt array, and phi
#                     tentative_cost = candidate.cost + d_candidate
#                     valid_parents.append(
#                         (candidate, tentative_cost, tmp_cost, tmp_phi))
#                 else:
#                     print("  -> REJECTED because some φ < 0.")
#             else:
#                 # If JXT fails, skip
#                 pass

#         if not valid_parents:
#             # No valid parent that satisfies j_limit AND φ>=0
#             continue

#         # Among valid parents, pick the one with the minimum path cost
#         valid_parents.sort(key=lambda x: x[1])
#         best_parent, best_cost, best_jxt, best_phi = valid_parents[0]

#         # Now finalize new_node with that best parent
#         new_node.parent = best_parent
#         new_node.cost = best_cost
#         d_parent = distance(best_parent, new_node)
#         new_node.time = best_parent.time + (d_parent / speed)
#         new_node.jxt_cost = best_jxt
#         new_node.phi = best_phi

#         # Add the new node to the tree
#         nodes.append(new_node)

#         # Rewiring: see if connecting near_nodes to new_node would improve them
#         for near_node in near_nodes:
#             if near_node is new_node:
#                 continue
#             d_rewire = distance(new_node, near_node)
#             potential_cost = new_node.cost + d_rewire
#             if potential_cost < near_node.cost:
#                 # Calculate new jxt cost from new_node to near_node
#                 tmp_rewire_node = Node(near_node.x, near_node.y)
#                 tmp_rewire_node.time = new_node.time + (d_rewire / speed)
#                 tmp_rewire_jxt = calc_new_jxt_cost(new_node, tmp_rewire_node,
#                                                    gamma, dt, segments)
#                 if all(jc <= j_limit for jc in tmp_rewire_jxt):
#                     # Also check φ >= 0
#                     tmp_rewire_node.jxt_cost = tmp_rewire_jxt
#                     tmp_rewire_node.pxq_values = near_node.pxq_values
#                     tmp_phi = compute_phi_values(new_node,
#                                                  tmp_rewire_node,
#                                                  segments,
#                                                  j_limit,
#                                                  gamma=gamma,
#                                                  alpha1=9,
#                                                  alpha2=9)
#                     # print(
#                     #     f"\nRewire φ check => new_node({new_node.x:.2f},{new_node.y:.2f}),"
#                     #     f" near_node({near_node.x:.2f},{near_node.y:.2f}):\n  φ-values = {[f'{pval:.4f}' for pval in tmp_phi]}"
#                     # )

#                     if all(pval >= 0 for pval in tmp_phi):
#                         # Rewire
#                         near_node.parent = new_node
#                         near_node.cost = potential_cost
#                         near_node.time = tmp_rewire_node.time
#                         near_node.jxt_cost = tmp_rewire_jxt
#                         near_node.phi = tmp_phi
#                     else:
#                         print("  -> Rewire REJECTED because some φ < 0.")

#         print(f"Iter: {i}, number of nodes: {len(nodes)}")

#         # Check for path to goal within threshold
#         current_path = get_path_to_goal(nodes,
#                                         goal,
#                                         threshold=threshold,
#                                         segments=segments,
#                                         gamma=gamma,
#                                         dt=dt,
#                                         j_limit=j_limit)
#         if current_path is not None:
#             # Compute length
#             current_length = compute_path_length([(n.x, n.y)
#                                                   for n in current_path])
#             if current_length < best_path_length:
#                 best_path_length = current_length
#                 path_progress.append((best_path_length, len(nodes)))

#     return nodes, path_progress

# def get_path_to_goal(nodes,
#                      goal,
#                      threshold=1.0,
#                      segments=None,
#                      gamma=0.9,
#                      dt=0.1,
#                      j_limit=15):
#     """
#     Extracts a path from the start to a node near the goal (within threshold),
#     then tries to add the exact goal if it doesn't break the JXT limit
#     AND passes φ ≥ 0 as well.

#     Returns a list of Node objects representing the path if found, else None.
#     """
#     # Find all nodes that are within 'threshold' of the goal
#     goal_candidates = []
#     for node in nodes:
#         if math.hypot(node.x - goal.x, node.y - goal.y) <= threshold:
#             goal_candidates.append(node)

#     if not goal_candidates:
#         return None

#     # Among candidates, pick the one with the smallest 'cost'
#     best_node = min(goal_candidates, key=lambda n: n.cost)

#     # Backtrack to get the path
#     path = []
#     node = best_node
#     while node is not None:
#         path.append(node)
#         node = node.parent
#     path.reverse()

#     # If the final node isn't exactly at (goal.x, goal.y), try to add an exact goal node
#     last = path[-1]
#     if (abs(last.x - goal.x) > 1e-9) or (abs(last.y - goal.y) > 1e-9):
#         # Proposed goal node
#         d = math.hypot(goal.x - last.x, goal.y - last.y)
#         speed = 0.1
#         goal_node = Node(goal.x, goal.y)
#         goal_node.time = last.time + (d / speed)
#         goal_node.parent = last

#         # Compute pxq and jxt for the final step
#         goal_node.pxq_values = compute_pxq_values(goal_node, segments)
#         goal_node_jxt = calc_new_jxt_cost(last, goal_node, gamma, dt, segments)

#         # Check if it violates the j_limit
#         if all(jc <= j_limit for jc in goal_node_jxt):
#             # Also check φ >= 0 for the final node
#             goal_node.jxt_cost = goal_node_jxt
#             goal_node_phi = compute_phi_values(last,
#                                                goal_node,
#                                                segments,
#                                                j_limit,
#                                                gamma=gamma,
#                                                alpha1=9,
#                                                alpha2=9)
#             # print(
#             #     f"\nGoal-node φ check => last({last.x:.2f},{last.y:.2f}),"
#             #     f" goal({goal.x:.2f},{goal.y:.2f}):\n  φ-values = {[f'{pval:.4f}' for pval in goal_node_phi]}"
#             # )
#             if all(pval >= 0 for pval in goal_node_phi):
#                 goal_node.phi = goal_node_phi
#                 path.append(goal_node)
#             else:
#                 print("  -> Final node REJECTED because some φ < 0.")
#         else:
#             # If the direct step to the exact goal breaks j_limit,
#             pass

#     return path

# def compute_path_length(path):
#     """
#     Given a list of (x,y) tuples, computes the total path length.
#     """
#     length = 0.0
#     for i in range(1, len(path)):
#         length += math.hypot(path[i][0] - path[i - 1][0],
#                              path[i][1] - path[i - 1][1])
#     return length

# # ------------------------------
# # NEW FUNCTION: CALCULATE TOTAL JXT COST FOR PATH
# # ------------------------------
# def calculate_total_jxt_for_path(path, segments, gamma=0.9, dt=0.1):
#     """
#     Given a path (list of Node objects), this function re-computes the
#     cumulative jxt cost from the very start to the end by chaining calls
#     to calc_new_jxt_cost. Useful for verifying the final path's cost.
#     """
#     if len(path) < 2:
#         return None

#     # Start cost from the first step
#     cumulative_jxt = calc_new_jxt_cost(path[0], path[1], gamma, dt, segments)
#     for i in range(2, len(path)):
#         # Make a "dummy" parent carrying the previously computed cost
#         dummy_parent = Node(path[i - 1].x, path[i - 1].y)
#         dummy_parent.time = path[i - 1].time
#         dummy_parent.jxt_cost = cumulative_jxt
#         # Accumulate
#         cumulative_jxt = calc_new_jxt_cost(dummy_parent, path[i], gamma, dt,
#                                            segments)

#     return cumulative_jxt

# # ------------------------------
# # PLOTTING
# # ------------------------------
# def plot_rrt(nodes, path, obstacles, start, goal, search_area):
#     plt.figure(figsize=(8, 8))
#     # Draw edges
#     for node in nodes:
#         if node.parent is not None:
#             plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")

#     # Final path in red
#     if path is not None and len(path) > 1:
#         px = [p.x for p in path]
#         py = [p.y for p in path]
#         plt.plot(px, py, "-r", linewidth=2, label="Final path")

#     # Obstacles
#     for obs in obstacles:
#         rect = plt.Rectangle((obs[0], obs[1]),
#                              obs[2] - obs[0],
#                              obs[3] - obs[1],
#                              color="gray",
#                              alpha=0.7)
#         plt.gca().add_patch(rect)

#     # Start and goal markers
#     plt.plot(start.x, start.y, "bs", markersize=10, label="Start")
#     plt.plot(goal.x, goal.y, "ms", markersize=10, label="Goal")

#     # Plot setup
#     plt.xlim(search_area[0], search_area[1])
#     plt.ylim(search_area[2], search_area[3])
#     plt.title("RRT* Path Planning with JXT & φ≥0 Constraints")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # ------------------------------
# # MAIN
# # ------------------------------
# def main():
#     # Example usage with a certain search area, start, goal, obstacles
#     search_area = (-5, 30, -5, 20)
#     start = Node(5.5, 1.0)
#     goal = Node(22.5, 10)
#     obstacles = []
#     obstacles.append((7.4, 2.3, 9.3, 3.8))
#     obstacles.append((10.1, 2.3, 10.9, 4.5))
#     obstacles.append((11.8, 2.5, 13.9, 4.9))
#     obstacles.append((14.5, 3.7, 17.0, 4.9))
#     obstacles.append((14.6, 2.1, 17.1, 3.3))
#     obstacles.append((17.8, 4.0, 20.1, 5.2))
#     obstacles.append((20.7, 2.3, 23.3, 3.6))
#     obstacles.append((23.5, 0.0, 26.1, 1.9))
#     obstacles.append((21.0, 6.2, 21.7, 7.1))
#     obstacles.append((22.4, 6.2, 23.1, 7.1))
#     obstacles.append((24.9, 7.6, 25.8, 9.2))
#     obstacles.append((24.9, 9.6, 26.0, 11.2))
#     obstacles.append((24.9, 13.6, 25.8, 14.5))
#     obstacles.append((19.9, 11.8, 21.6, 14.1))
#     obstacles.append((14.7, 11.4, 15.8, 12.8))
#     obstacles.append((9.6, 10.8, 10.2, 12.3))
#     obstacles.append((7.1, 10.8, 7.7, 12.3))
#     obstacles.append((0.0, 6.1, 2.2, 12.2))
#     obstacles.append((15.6, 6.2, 17.0, 9.8))
#     obstacles.append((17.6, 8.3, 20.8, 11.6))
#     obstacles.append((17.6, 6.3, 19.7, 8.1))
#     obstacles.append((11.9, 6.0, 14.0, 8.3))
#     obstacles.append((2.2, 8.4, 5.1, 10.3))
#     obstacles.append((5.1, 6.4, 7.8, 9.9))
#     obstacles.append((7.8, 8.0, 10.8, 9.7))

#     max_iter = 3500
#     step_size = 1
#     goal_sample_rate = 0.2
#     neighbor_radius = 2
#     threshold = 1.0
#     j_limit = 15  # Example j_limit value

#     # Run RRT*
#     nodes, path_progress = rrt_star(start,
#                                     goal,
#                                     obstacles,
#                                     search_area,
#                                     max_iter,
#                                     step_size,
#                                     goal_sample_rate,
#                                     neighbor_radius,
#                                     threshold,
#                                     gamma=0.9,
#                                     dt=0.1,
#                                     j_limit=j_limit)

#     # Extract final path
#     final_path = get_path_to_goal(nodes,
#                                   goal,
#                                   threshold=threshold,
#                                   segments=get_obstacle_segments(obstacles),
#                                   gamma=0.9,
#                                   dt=0.1,
#                                   j_limit=j_limit)

#     print("\nPath Length Samples (best path found over iterations):")
#     print(path_progress)

#     if final_path is not None:
#         # Check if each node in final path respects j_limit
#         all_within_limit = True
#         for node in final_path:
#             if node.jxt_cost is not None and any(jxt > j_limit
#                                                  for jxt in node.jxt_cost):
#                 all_within_limit = False
#                 break

#         if all_within_limit:
#             print(
#                 "\nAll nodes in the final path satisfy the j_limit constraint."
#             )
#         else:
#             print("\nWARNING: Some node(s) in the final path exceed j_limit!")

#         print("\nFinal Path (Node, Time, pxq Values, jxt_cost, phi):")
#         for node in final_path:
#             print(f"Node: ({node.x:.2f}, {node.y:.2f}), Time: {node.time:.4f}")
#             if node.pxq_values is not None:
#                 print("  pxq values:",
#                       [f"{val:.4f}" for val in node.pxq_values])
#             if node.jxt_cost is not None:
#                 print("  jxt_cost:", [f"{val:.4f}" for val in node.jxt_cost])
#             if node.phi is not None:
#                 print("  phi:", [f"{pval:.4f}" for pval in node.phi])

#         total_jxt = calculate_total_jxt_for_path(
#             final_path, get_obstacle_segments(obstacles), gamma=0.9, dt=0.1)
#         print("\nTotal jxt cost for the final path:")
#         if total_jxt:
#             print([f"{val:.4f}" for val in total_jxt])
#         else:
#             print("No path or single-node path (nothing to compute).")

#     else:
#         print("\nNo path found that satisfies j_limit (and φ≥0)!")

#     # Visualize
#     plot_rrt(nodes, final_path, obstacles, start, goal, search_area)

# if __name__ == "__main__":
#     main()
