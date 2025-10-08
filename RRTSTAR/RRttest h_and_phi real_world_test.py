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
        self.pxq_values = None  # pxq for the "noise" obstacles only
        self.jxt_cost = None  # Jₓₜ cost values (one per "noise" obstacle segment)
        self.phi = None  # φ values (one per "noise" obstacle segment)


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


def get_obstacle_segments(obstacles_list):
    """
    Given obstacles as (xmin, ymin, xmax, ymax),
    returns a list of segments (each as ((x1,y1),(x2,y2))).
    Each obstacle contributes four segments.
    """
    segments = []
    for obs in obstacles_list:
        xmin, ymin, xmax, ymax = obs
        segments.append(((xmin, ymin), (xmax, ymin)))
        segments.append(((xmax, ymin), (xmax, ymax)))
        segments.append(((xmax, ymax), (xmin, ymax)))
        segments.append(((xmin, ymax), (xmin, ymin)))
    return segments


def compute_pxq_values(node, segments):
    """
    For a given node and list of "noise" segments, computes pxq = (a*d + b)^(-2) + c + 60,
    where d=dist(node, segment), a=0.4578, b=0.1593, c=-41.0.
    """
    a = 0.4578
    b = 0.1593
    c = -40.500
    #c = -41.00
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
    total = 0.0
    for i in range(1, len(gaussian)):
        total += ((gaussian[i] + gaussian[i - 1]) / 2.0) * dt
    return total


def calc_new_jxt_cost(from_node, to_node, gamma=0.9, dt=0.1, segments=None):
    """
    Computes the new jxt_cost for to_node given its parent (from_node), using:
      J(q,t₂) = γ^(t₂-t₁)*J(q,t₁) + ∫ₜ₁ᵗ₂ γ^(t₂−τ)* p(q,x(τ)) dτ.
    'segments' are the noise obstacle segments used for pxq.
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
        if abs(t2 - t1) < 1e-9:
            x_tau = from_node.x
            y_tau = from_node.y
        else:
            alpha = (tau - t1) / (t2 - t1)
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

    # Compute pxq values for the newly created node (noise obstacles only)
    new_node.pxq_values = compute_pxq_values(new_node, segments)
    return new_node


def is_inside_box(point, box):
    """
    Checks if point (x,y) is inside the given box (xmin, ymin, xmax, ymax).
    """
    x, y = point
    return (box[0] <= x <= box[2]) and (box[1] <= y <= box[3])


def collision_check(node1, node2, obstacles_for_collision):
    """
    Checks if the straight-line path between node1 and node2 is collision-free
    by sampling points. 'obstacles_for_collision' are the obstacles we
    consider for collisions only (the union of noise obstacles + second obstacles).
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
        for obs in obstacles_for_collision:
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
                       noise_segments,
                       j_limit,
                       gamma=0.9,
                       alpha1=0.9,
                       alpha2=0.9,
                       a=0.4578,
                       b=0.1593):
    """
    Computes φ_{J,2} for each noise segment:
      φ_j =
        -log(γ)*(log(γ)*J_j + p_j)
        + 2*a*(a*d + b)^(-3)*[((q - x)^T / d)*u]
        + α1[ -log(γ)*J_j - p_j ]
        + α2[ -log(γ)*J_j - p_j + α1(J_limit - J_j) ]
    """
    phi_list = []
    delta_t = to_node.time - from_node.time
    # Velocity vector u
    if abs(delta_t) < 1e-9:
        u = np.array([0.0, 0.0])
    else:
        u = np.array([to_node.x - from_node.x, to_node.y - from_node.y
                      ]) / delta_t

    for j, seg in enumerate(noise_segments):
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

        second_term = 2.0 * a * ((a * dist_j + b)**-3) * dot_val

        part1 = -math.log(gamma) * (math.log(gamma) * J_j + p_j)
        part3 = alpha1 * (-math.log(gamma) * J_j - p_j)
        part4 = alpha2 * (-math.log(gamma) * J_j - p_j + alpha1 *
                          (j_limit - J_j))

        phi_j = part1 + second_term + part3 + part4
        phi_list.append(phi_j)

    return phi_list


# ------------------------------
# GET PATH TO GOAL
# ------------------------------
def get_path_to_goal(nodes,
                     goal,
                     threshold=0.1,
                     noise_segments=None,
                     gamma=0.9,
                     dt=0.1,
                     j_limit=195.0,
                     obstacles_for_collision=None):
    """
    Extracts a path from the start to a node near the goal (within threshold).
    If that node is not exactly the goal, try to connect to the goal (if collision-free, etc).
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
        if obstacles_for_collision is not None:
            goal_node = Node(goal.x, goal.y)
            dist_to_goal = math.hypot(goal.x - last.x, goal.y - last.y)
            speed = 0.1
            goal_node.time = last.time + (dist_to_goal / speed)
            goal_node.parent = last
            goal_node.pxq_values = compute_pxq_values(goal_node,
                                                      noise_segments)

            # Collision check with obstacles_for_collision
            if collision_check(last, goal_node, obstacles_for_collision):
                # JXT cost
                goal_node_jxt = calc_new_jxt_cost(last, goal_node, gamma, dt,
                                                  noise_segments)
                if all(jc <= j_limit for jc in goal_node_jxt):
                    goal_node.jxt_cost = goal_node_jxt
                    # Check φ ≥ 0
                    goal_node_phi = compute_phi_values(last,
                                                       goal_node,
                                                       noise_segments,
                                                       j_limit,
                                                       gamma=gamma,
                                                       alpha1=0.9,
                                                       alpha2=0.9)
                    if all(pval >= 0 for pval in goal_node_phi):
                        goal_node.phi = goal_node_phi
                        path.append(goal_node)
                    else:
                        print("  -> Final node REJECTED: some φ < 0.")
                else:
                    print("  -> Final node REJECTED: jxt_cost > j_limit.")
            else:
                print("  -> Final node REJECTED: collision with obstacle.")

    return path


# ------------------------------
# RRT* ALGORITHM
# ------------------------------
def rrt_star(start,
             goal,
             obstacles_noise,
             obstacles_collision,
             search_area,
             max_iter=500,
             step_size=1,
             goal_sample_rate=0.2,
             neighbor_radius=2,
             threshold=0.1,
             gamma=0.9,
             dt=0.1,
             j_limit=195.0):
    """
    RRT* with:
      - obstacles_noise: used for pxq & jxt cost
      - obstacles_collision: used for collision only
      - jxt <= j_limit, φ >= 0 constraints
    Returns (nodes, path_progress).
    """
    noise_segments = get_obstacle_segments(obstacles_noise)

    # Initialize the start node
    start.pxq_values = compute_pxq_values(start, noise_segments)
    start.jxt_cost = [0.0 for _ in range(len(noise_segments))]
    start.phi = [0.0 for _ in range(len(noise_segments))]
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

        # Steer
        nearest_node = get_nearest_node(nodes, sample)
        new_node = steer(nearest_node, sample, step_size, noise_segments)

        # Collision check from nearest_node to new_node
        if not collision_check(nearest_node, new_node, obstacles_collision):
            continue

        # Find near nodes
        near_nodes = []
        for node in nodes:
            if distance(node, new_node) <= neighbor_radius:
                if collision_check(node, new_node, obstacles_collision):
                    near_nodes.append(node)

        # Include nearest_node if not present
        if nearest_node not in near_nodes:
            near_nodes.append(nearest_node)

        valid_parents = []
        for candidate in near_nodes:
            d_candidate = distance(candidate, new_node)
            travel_time = d_candidate / speed

            # Temporarily set times
            tmp_node = Node(new_node.x, new_node.y)
            tmp_node.time = candidate.time + travel_time

            tmp_cost = calc_new_jxt_cost(candidate, tmp_node, gamma, dt,
                                         noise_segments)
            if all(jc <= j_limit for jc in tmp_cost):
                # Check φ
                tmp_node.jxt_cost = tmp_cost
                tmp_node.pxq_values = new_node.pxq_values
                tmp_phi = compute_phi_values(candidate,
                                             tmp_node,
                                             noise_segments,
                                             j_limit,
                                             gamma=gamma,
                                             alpha1=0.9,
                                             alpha2=0.9)
                if all(pval >= 0 for pval in tmp_phi):
                    tentative_cost = candidate.cost + d_candidate
                    valid_parents.append(
                        (candidate, tentative_cost, tmp_cost, tmp_phi))

        if not valid_parents:
            continue

        valid_parents.sort(key=lambda x: x[1])
        best_parent, best_cost, best_jxt, best_phi = valid_parents[0]

        new_node.parent = best_parent
        new_node.cost = best_cost
        d_parent = distance(best_parent, new_node)
        new_node.time = best_parent.time + (d_parent / speed)
        new_node.jxt_cost = best_jxt
        new_node.phi = best_phi

        nodes.append(new_node)

        # Rewiring
        for near_node in near_nodes:
            if near_node is new_node:
                continue
            d_rewire = distance(new_node, near_node)
            potential_cost = new_node.cost + d_rewire
            if potential_cost < near_node.cost:
                # Check collision
                if collision_check(new_node, near_node, obstacles_collision):
                    tmp_rewire_node = Node(near_node.x, near_node.y)
                    tmp_rewire_node.time = new_node.time + (d_rewire / speed)
                    tmp_rewire_jxt = calc_new_jxt_cost(new_node,
                                                       tmp_rewire_node, gamma,
                                                       dt, noise_segments)
                    if all(jc <= j_limit for jc in tmp_rewire_jxt):
                        tmp_rewire_node.jxt_cost = tmp_rewire_jxt
                        tmp_rewire_node.pxq_values = near_node.pxq_values
                        tmp_phi = compute_phi_values(new_node,
                                                     tmp_rewire_node,
                                                     noise_segments,
                                                     j_limit,
                                                     gamma=gamma,
                                                     alpha1=0.9,
                                                     alpha2=0.9)
                        if all(pval >= 0 for pval in tmp_phi):
                            near_node.parent = new_node
                            near_node.cost = potential_cost
                            near_node.time = tmp_rewire_node.time
                            near_node.jxt_cost = tmp_rewire_jxt
                            near_node.phi = tmp_phi

        # Check path
        current_path = get_path_to_goal(
            nodes,
            goal,
            threshold=threshold,
            noise_segments=noise_segments,
            gamma=gamma,
            dt=dt,
            j_limit=j_limit,
            obstacles_for_collision=obstacles_collision)
        if current_path is not None:
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


def calculate_total_jxt_for_path(path, noise_segments, gamma=0.9, dt=0.1):
    """
    Re-compute cumulative jxt for a path from start to end by chaining calc_new_jxt_cost.
    """
    if len(path) < 2:
        return None

    cumulative_jxt = calc_new_jxt_cost(path[0], path[1], gamma, dt,
                                       noise_segments)
    for i in range(2, len(path)):
        dummy_parent = Node(path[i - 1].x, path[i - 1].y)
        dummy_parent.time = path[i - 1].time
        dummy_parent.jxt_cost = cumulative_jxt
        cumulative_jxt = calc_new_jxt_cost(dummy_parent, path[i], gamma, dt,
                                           noise_segments)

    return cumulative_jxt


def plot_rrt(nodes, path, obstacles_noise, obstacles_2, start, goal,
             search_area):
    """
    Plots the RRT (green edges), the final path (red), and both sets of obstacles.
      - obstacles_noise: used for pxq/noise constraints
      - obstacles_2: used for collision only
    """
    plt.figure(figsize=(8, 8))
    # RRT edges
    for node in nodes:
        if node.parent is not None:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")

    # Final path in red
    if path is not None and len(path) > 1:
        px = [p.x for p in path]
        py = [p.y for p in path]
        plt.plot(px, py, "-r", linewidth=2, label="Final Path")

    # Plot noise obstacles (gray)
    for obs in obstacles_noise:
        rect = plt.Rectangle((obs[0], obs[1]),
                             obs[2] - obs[0],
                             obs[3] - obs[1],
                             color="gray",
                             alpha=0.7,
                             label="Noise Obstacles")
        plt.gca().add_patch(rect)

    # Plot collision-only obstacles (blueish)
    for obs in obstacles_2:
        rect2 = plt.Rectangle((obs[0], obs[1]),
                              obs[2] - obs[0],
                              obs[3] - obs[1],
                              color="blue",
                              alpha=0.4,
                              label="Collision-Only Obstacles")
        plt.gca().add_patch(rect2)

    # Start & Goal
    #plt.plot(start.x, start.y, "bs", markersize=10, label="Start")
    plt.plot(goal.x, goal.y, "ms", markersize=10, label="Goal")
    plt.scatter([-0.375, -0.385], [5.474, 6.168], s=100, c='purple')
    plt.xlim(search_area[0], search_area[1])
    plt.ylim(search_area[2], search_area[3])
    #plt.title("RRT* with Noise Obstacles + Separate Collision-Only Obstacles")
    plt.xlabel("X")
    plt.ylabel("Y")
    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)
    plt.show()


# ------------------------------
# MAIN (Example)
# ------------------------------
def main():
    # Example area
    search_area = (-1, 3, 1, 9)
    start = Node(-0.3, 2.3)
    goal = Node(-0.3, 7.95)

    # 1) Obstacles used for NOISE (i.e. pxq, jxt).  ("obstacles_noise")
    obstacles_noise = [
        (-1.0, 5.474, -0.385, 6.168),  # Speaker or something
    ]

    # 2) Obstacles used ONLY for collision.  ("obstacles_2")
    #    i.e. these do NOT appear in pxq/jxt calculations.
    obstacles_2 = [
        (0.25, 5.455, 0.57, 5.825),  # Box1
        (1.41, 5.455, 1.73, 5.825),  # Box2
    ]

    # For actual collision checking, let's unify both sets:
    obstacles_collision = obstacles_noise + obstacles_2

    max_iter = 2500
    step_size = 1.0
    goal_sample_rate = 0.2
    neighbor_radius = 1.0
    threshold = 0.1
    j_limit = 195

    # Run RRT*
    nodes, path_progress = rrt_star(start,
                                    goal,
                                    obstacles_noise,
                                    obstacles_collision,
                                    search_area,
                                    max_iter=max_iter,
                                    step_size=step_size,
                                    goal_sample_rate=goal_sample_rate,
                                    neighbor_radius=neighbor_radius,
                                    threshold=threshold,
                                    gamma=0.9,
                                    dt=0.1,
                                    j_limit=j_limit)

    # Attempt final path to the goal
    final_path = get_path_to_goal(
        nodes,
        goal,
        threshold=threshold,
        noise_segments=get_obstacle_segments(obstacles_noise),
        gamma=0.9,
        dt=0.1,
        j_limit=j_limit,
        obstacles_for_collision=obstacles_collision)

    print("\nPath Length Samples (best path over iterations):")
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

        # Compute total jxt over the path
        noise_segments = get_obstacle_segments(obstacles_noise)
        total_jxt = calculate_total_jxt_for_path(final_path,
                                                 noise_segments,
                                                 gamma=0.9,
                                                 dt=0.1)
        print("\nTotal jxt cost for the final path:")
        if total_jxt:
            print([f"{val:.4f}" for val in total_jxt])
        else:
            print("No path or single-node path (nothing to compute).")
    else:
        print(
            "\nNo path found that meets j_limit & φ≥0 all the way to the goal."
        )

    # Visualize
    plot_rrt(nodes, final_path, obstacles_noise, obstacles_2, start, goal,
             search_area)

    if final_path is not None and len(final_path) > 0:
        # Each obstacle in get_obstacle_segments(...) yields 4 segments in order:
        #   Index 0 = bottom edge
        #   Index 1 = right edge
        #   Index 2 = top edge
        #   Index 3 = left edge
        #
        # So the "right side" of the first obstacle is segment index 1.

        # Make sure the final path actually has jxt_cost
        segment_index = 1  # Right side of the *first* obstacle
        times = [node.time for node in final_path]

        # Double-check that each node has enough segments in jxt_cost
        if all(len(node.jxt_cost) > segment_index for node in final_path):
            # Gather the jxt values for that single segment
            jxt_right_edge = [
                node.jxt_cost[segment_index] for node in final_path
            ]

            plt.figure()
            plt.plot(times,
                     jxt_right_edge,
                     color='blue',
                     linewidth=2.0,
                     label='J(q,t) vs Time')
            plt.axhline(y=j_limit,
                        color='r',
                        linestyle='--',
                        label='J_limit = 195')
            plt.xlabel('Time')
            plt.ylabel('J(q,t)')
            #plt.title()
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Not all nodes have enough JXT segments for index 1!")


if __name__ == "__main__":
    main()
