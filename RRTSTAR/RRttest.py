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
        self.pxq_values = None  # List of pxq values for each obstacle segment (computed once)
        self.jxt_cost = None  # List of Jₓₜ cost values (one per segment)


# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def distance(n1, n2):
    return math.hypot(n1.x - n2.x, n1.y - n2.y)


def point_to_segment_distance(point, seg_start, seg_end):
    """
    Computes the minimum distance from a point to a line segment.
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
        return math.hypot(apx, apy)
    t = (apx * abx + apy * aby) / ab_squared
    t = max(0, min(1, t))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)


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
    For each segment, using the formula:
      pxq = (a * d + b)^(-2) + c + 60,
    where d is the distance from the node to the segment,
          a = 0.4578, b = 0.1593, c = -41.00.
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
    Computes the new jxt_cost for to_node given its parent (from_node)
    using the recursive formula:
      J(q,t₂) = γ^(t₂-t₁) * J(q,t₁) + ∫ₜ₁ᵗ₂ γ^(t₂ - τ) p(q, x(τ)) dτ.
    This is done for each obstacle segment.
    
    We discretize the time interval [from_node.time, to_node.time] with step dt.
    """
    deltaT = to_node.time - from_node.time
    deltaX = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
    if from_node.jxt_cost is None:
        parent_jxt = [0.0 for _ in range(len(segments))]
    else:
        parent_jxt = from_node.jxt_cost

    # First term: gamma^(deltaT)*parent_jxt
    first_term = [(gamma**deltaT) * cost for cost in parent_jxt]

    # If there's no time difference, we just return the scaled parent cost
    if abs(deltaT) < 1e-9:
        return first_term

    # Prepare for integral
    #n_steps = max(1, int(deltaT / dt))
    n_steps = max(1, int(deltaT / dt))
    tau_values = [from_node.time + i * dt for i in range(n_steps)]
    if tau_values[-1] < to_node.time:
        tau_values.append(to_node.time)

    t1 = from_node.time
    t2 = to_node.time
    f_values_per_seg = [[] for _ in range(len(segments))]
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    dist = math.hypot(dx, dy)

    for tau in tau_values:
        #alpha = (tau - t1) / deltaT if deltaT > 0 else 0
        alpha = (tau - t1) / deltaT if deltaT > 0 else 0
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
    Computes the new node’s cost, time, and its pxq_values.
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
    Checks if the straight-line path between node1 and node2 is collision-free.
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
             threshold=0.3,
             gamma=0.9,
             dt=0.1,
             j_limit=7.5):
    """
    Runs the RRT* algorithm with a JXT cost constraint.
    Returns:
      - nodes: all nodes in the tree.
      - path_progress: list of (best_path_length, #nodes_at_that_time).
    """
    segments = get_obstacle_segments(obstacles)

    # Initialize the start node
    start.pxq_values = compute_pxq_values(start, segments)
    start.jxt_cost = [0.0 for _ in range(len(segments))]

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

        # Collision check
        if not collision_check(nearest_node, new_node, obstacles):
            # Skipping due to collision
            continue

        # Find near nodes for potential re-parenting
        near_nodes = []
        for node in nodes:
            if distance(node, new_node) <= neighbor_radius:
                if collision_check(node, new_node, obstacles):
                    near_nodes.append(node)

        # Try to find a valid parent among near_nodes (plus the nearest_node itself)
        if nearest_node not in near_nodes:
            near_nodes.append(nearest_node)

        valid_parents = []
        for candidate in near_nodes:
            # Compute the new jxt cost if 'new_node' is connected to 'candidate'
            d_candidate = distance(candidate, new_node)
            travel_time = d_candidate / speed

            # Temporarily set 'to_node' times for the cost calculation
            tmp_node = Node(new_node.x, new_node.y)
            tmp_node.time = candidate.time + travel_time  # what it would be if parent is candidate
            tmp_cost = calc_new_jxt_cost(candidate, tmp_node, gamma, dt,
                                         segments)

            # Check JXT cost limit
            if all(jc <= j_limit for jc in tmp_cost):
                # If valid, record the parent, the cost, and the jxt array
                tentative_cost = candidate.cost + d_candidate
                valid_parents.append((candidate, tentative_cost, tmp_cost))

        if not valid_parents:
            # No valid parent that satisfies j_limit
            continue

        # Among valid parents, pick the one with the minimum path cost
        valid_parents.sort(key=lambda x: x[1])
        best_parent, best_cost, best_jxt = valid_parents[0]

        # Now finalize new_node with that best parent
        new_node.parent = best_parent
        new_node.cost = best_cost
        d_parent = distance(best_parent, new_node)
        new_node.time = best_parent.time + (d_parent / speed)
        new_node.jxt_cost = best_jxt

        # Add the new node to the tree
        nodes.append(new_node)

        # Rewiring: see if connecting near_nodes to new_node would improve them
        for near_node in near_nodes:
            if near_node is new_node:
                continue
            d_rewire = distance(new_node, near_node)
            potential_cost = new_node.cost + d_rewire
            if potential_cost < near_node.cost:
                # Calculate new jxt cost from new_node to near_node
                tmp_rewire_node = Node(near_node.x, near_node.y)
                tmp_rewire_node.time = new_node.time + (d_rewire / speed)
                tmp_rewire_jxt = calc_new_jxt_cost(new_node, tmp_rewire_node,
                                                   gamma, dt, segments)
                if all(jc <= j_limit for jc in tmp_rewire_jxt):
                    # Rewire
                    near_node.parent = new_node
                    near_node.cost = potential_cost
                    near_node.time = tmp_rewire_node.time
                    near_node.jxt_cost = tmp_rewire_jxt
        print(f"Iter: {i}, number of nodes: {len(nodes)}")
        # Check for path to goal within threshold
        current_path = get_path_to_goal(nodes,
                                        goal,
                                        threshold=threshold,
                                        segments=segments,
                                        gamma=gamma,
                                        dt=dt,
                                        j_limit=j_limit)
        if current_path is not None:
            # Compute length
            current_length = compute_path_length([(n.x, n.y)
                                                  for n in current_path])
            if current_length < best_path_length:
                best_path_length = current_length
                path_progress.append((best_path_length, len(nodes)))

    return nodes, path_progress


def get_path_to_goal(nodes,
                     goal,
                     threshold=0.3,
                     segments=None,
                     gamma=0.9,
                     dt=0.1,
                     j_limit=7.5):
    """
    Extracts a path from the start to a node near the goal (within threshold),
    then tries to add the exact goal if it doesn't break the JXT limit.

    Returns a list of Node objects representing the path if found, else None.
    """
    # Find all nodes that are within 'threshold' of the goal
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
    node = best_node
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()

    # If the final node isn't exactly at (goal.x, goal.y), try to add an exact goal node
    last = path[-1]
    if (abs(last.x - goal.x) > 1e-9) or (abs(last.y - goal.y) > 1e-9):
        # Proposed goal node
        d = math.hypot(goal.x - last.x, goal.y - last.y)
        speed = 0.1
        goal_node = Node(goal.x, goal.y)
        goal_node.time = last.time + (d / speed)
        goal_node.parent = last

        # Compute pxq and jxt for the final step
        goal_node.pxq_values = compute_pxq_values(goal_node, segments)
        goal_node_jxt = calc_new_jxt_cost(last, goal_node, gamma, dt, segments)

        # Check if it violates the j_limit
        if all(jc <= j_limit for jc in goal_node_jxt):
            goal_node.jxt_cost = goal_node_jxt
            path.append(goal_node)
        else:
            # If the direct step to the exact goal breaks j_limit,
            # we just return the path up to 'best_node'
            pass

    return path


def compute_path_length(path):
    """
    Given a list of (x,y) tuples, computes the total path length.
    """
    length = 0.0
    for i in range(1, len(path)):
        length += math.hypot(path[i][0] - path[i - 1][0],
                             path[i][1] - path[i - 1][1])
    return length


# ------------------------------
# NEW FUNCTION: CALCULATE TOTAL JXT COST FOR PATH
# ------------------------------
def calculate_total_jxt_for_path(path, segments, gamma=0.9, dt=0.1):
    """
    Given a path (list of Node objects), this function re-computes the
    cumulative jxt cost from the very start to the end by chaining calls
    to calc_new_jxt_cost. Useful for verifying the final path's cost.
    """
    if len(path) < 2:
        return None

    # Start cost from the first step
    cumulative_jxt = calc_new_jxt_cost(path[0], path[1], gamma, dt, segments)
    for i in range(2, len(path)):
        # Make a "dummy" parent carrying the previously computed cost
        dummy_parent = Node(path[i - 1].x, path[i - 1].y)
        dummy_parent.time = path[i - 1].time
        dummy_parent.jxt_cost = cumulative_jxt
        # Accumulate
        cumulative_jxt = calc_new_jxt_cost(dummy_parent, path[i], gamma, dt,
                                           segments)

    return cumulative_jxt


# ------------------------------
# PLOTTING
# ------------------------------
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

    # Plot setup
    plt.xlim(search_area[0], search_area[1])
    plt.ylim(search_area[2], search_area[3])
    plt.title("RRT* Path Planning with JXT Constraint")
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.legend()
    plt.grid(True)
    plt.show()


# ------------------------------
# MAIN
# ------------------------------
def main():
    # search_area = (xmin, xmax, ymin, ymax)

    # search_area = (0, 100, 0, 100)
    # start = Node(5, 5)
    # goal = Node(90, 90)
    search_area = (-5, 30, -5, 20)
    start = Node(5.5, 1.0)
    goal = Node(22.5, 10)
    obstacles = []
    # obstacles.append((40, 40, 60, 60))
    # obstacles.append((20, 70, 30, 80))
    # obstacles.append((70, 20, 80, 30))
    obstacles.append((7.4, 2.3, 9.3, 3.8))
    obstacles.append((10.1, 2.3, 10.9, 4.5))  # 720 Albany
    obstacles.append((11.8, 2.5, 13.9, 4.9))  # 710 Albany
    obstacles.append((14.5, 3.7, 17.0, 4.9))  # 700 Albany
    obstacles.append((14.6, 2.1, 17.1, 3.3))  # 670 Albany
    obstacles.append((17.8, 4.0, 20.1, 5.2))  # 650 Albany
    obstacles.append((20.7, 2.3, 23.3, 3.6))  # 620 Albany
    obstacles.append((23.5, 0.0, 26.1, 1.9))  # 610 Albany
    obstacles.append((21.0, 6.2, 21.7, 7.1))  # 615 Albany
    obstacles.append((22.4, 6.2, 23.1, 7.1))  # 609 Albany
    obstacles.append((24.9, 7.6, 25.8, 9.2))  # 100 E.Canton
    obstacles.append((24.9, 9.6, 26.0, 11.2))  # Employee Parking Lot
    obstacles.append((24.9, 13.6, 25.8, 14.5))  # 660 Harrison
    obstacles.append((19.9, 11.8, 21.6, 14.1))  # BMC DOCTORS
    obstacles.append((14.7, 11.4, 15.8, 12.8))  # VOSE
    obstacles.append((9.6, 10.8, 10.2, 12.3))  # BCD
    obstacles.append((7.1, 10.8, 7.7, 12.3))  # FGH
    obstacles.append((0.0, 6.1, 2.2, 12.2))  # NORTHHAMPTON SQUARE
    obstacles.append((15.6, 6.2, 17.0, 9.8))  # SOLOMON CARTER
    obstacles.append((17.6, 8.3, 20.8, 11.6))  # 88E NEWTON
    obstacles.append((17.6, 6.3, 19.7, 8.1))  # GOLDMAN SCHOOL OF DENTAL
    obstacles.append((11.9, 6.0, 14.0, 8.3))  # SCHOOL OF PUBLIC HEALTH
    obstacles.append((2.2, 8.4, 5.1, 10.3))  # BMC YAWKEY CENTER
    obstacles.append((5.1, 6.4, 7.8, 9.9))  # BMC MENINO
    obstacles.append((7.8, 8.0, 10.8, 9.7))  # BMC MOAKLEY
    # obstacles.append((-1.0, 5.474, -0.385, 6.168))  # Speaker Obstacle
    # obstacles.append((0.25, 5.455, 0.57, 5.825))  # Box 1
    # obstacles.append((1.41, 5.455, 1.73, 5.825))  # Box 2

    max_iter = 6000
    #step_size = 5
    step_size = 1
    goal_sample_rate = 0.2
    #neighbor_radius = 10
    neighbor_radius = 2.0
    threshold = 0.3
    j_limit = 197  # Example j_limit value

    # Run RRT*
    nodes, path_progress = rrt_star(start,
                                    goal,
                                    obstacles,
                                    search_area,
                                    max_iter,
                                    step_size,
                                    goal_sample_rate,
                                    neighbor_radius,
                                    threshold,
                                    gamma=0.9,
                                    dt=0.1,
                                    j_limit=j_limit)

    # Extract final path
    final_path = get_path_to_goal(nodes,
                                  goal,
                                  threshold=threshold,
                                  segments=get_obstacle_segments(obstacles),
                                  gamma=0.9,
                                  dt=0.1,
                                  j_limit=j_limit)

    print("\nPath Length Samples (best path found over iterations):")
    print(path_progress)

    if final_path is not None:
        # Check if each node in final path respects j_limit
        all_within_limit = True
        for node in final_path:
            if node.jxt_cost is not None and any(jxt > j_limit
                                                 for jxt in node.jxt_cost):
                all_within_limit = False
                break

        if all_within_limit:
            print(
                "\nAll nodes in the final path satisfy the j_limit constraint."
            )
        else:
            print("\nWARNING: Some node(s) in the final path exceed j_limit!")

        print("\nFinal Path (Node, Time, pxq Values, jxt_cost):")
        for node in final_path:
            print(f"Node: ({node.x:.2f}, {node.y:.2f}), Time: {node.time:.4f}")
            if node.pxq_values is not None:
                print("  pxq values:",
                      [f"{val:.4f}" for val in node.pxq_values])
            if node.jxt_cost is not None:
                print("  jxt_cost:", [f"{val:.4f}" for val in node.jxt_cost])

        total_jxt = calculate_total_jxt_for_path(
            final_path, get_obstacle_segments(obstacles), gamma=0.9, dt=0.1)
        print("\nTotal jxt cost for the final path:")
        if total_jxt:
            print([f"{val:.4f}" for val in total_jxt])
        else:
            print("No path or single-node path (nothing to compute).")

    else:
        print("\nNo path found that satisfies j_limit!")

    # Visualize
    plot_rrt(nodes, final_path, obstacles, start, goal, search_area)

    if final_path is not None and len(final_path) > 0:
        # Each box (obstacle) has 4 consecutive segments.
        # The first obstacle is at indices [0, 1, 2, 3].
        seg_indices = [0, 1, 2, 3]  # first obstacle's 4 edges
        j_limit = 7.5
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

#THIS ONE WORKS BUT ITS THE ORIGINAL CODE WITH INITIAL DIMENTSIONS

# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import random

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

# # ------------------------------
# # HELPER FUNCTIONS
# # ------------------------------
# def distance(n1, n2):
#     return math.hypot(n1.x - n2.x, n1.y - n2.y)

# def point_to_segment_distance(point, seg_start, seg_end):
#     """
#     Computes the minimum distance from a point to a line segment.
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
#         return math.hypot(apx, apy)
#     t = (apx * abx + apy * aby) / ab_squared
#     t = max(0, min(1, t))
#     closest_x = ax + t * abx
#     closest_y = ay + t * aby
#     return math.hypot(px - closest_x, py - closest_y)

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

#     We discretize the time interval [from_node.time, to_node.time] with step dt.
#     """
#     deltaT = to_node.time - from_node.time
#     if from_node.jxt_cost is None:
#         parent_jxt = [0.0 for _ in range(len(segments))]
#     else:
#         parent_jxt = from_node.jxt_cost

#     # First term: gamma^(deltaT)*parent_jxt
#     first_term = [(gamma**deltaT) * cost for cost in parent_jxt]

#     # If there's no time difference, we just return the scaled parent cost
#     if abs(deltaT) < 1e-9:
#         return first_term

#     # Prepare for integral
#     n_steps = max(1, int(deltaT / dt))
#     tau_values = [from_node.time + i * dt for i in range(n_steps)]
#     if tau_values[-1] < to_node.time:
#         tau_values.append(to_node.time)

#     t1 = from_node.time
#     t2 = to_node.time
#     f_values_per_seg = [[] for _ in range(len(segments))]
#     dx = to_node.x - from_node.x
#     dy = to_node.y - from_node.y
#     dist = math.hypot(dx, dy)

#     for tau in tau_values:
#         alpha = (tau - t1) / deltaT if deltaT > 0 else 0
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
#     resolution = 0.5
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
# # RRT* ALGORITHM
# # ------------------------------
# def rrt_star(start,
#              goal,
#              obstacles,
#              search_area,
#              max_iter=500,
#              step_size=5,
#              goal_sample_rate=0.1,
#              neighbor_radius=10,
#              threshold=5.0,
#              gamma=0.9,
#              dt=0.1,
#              j_limit=180.0):
#     """
#     Runs the RRT* algorithm with a JXT cost constraint.
#     Returns:
#       - nodes: all nodes in the tree.
#       - path_progress: list of (best_path_length, #nodes_at_that_time).
#     """
#     segments = get_obstacle_segments(obstacles)

#     # Initialize the start node
#     start.pxq_values = compute_pxq_values(start, segments)
#     start.jxt_cost = [0.0 for _ in range(len(segments))]

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
#             tmp_node.time = candidate.time + travel_time  # what it would be if parent is candidate
#             tmp_cost = calc_new_jxt_cost(candidate, tmp_node, gamma, dt,
#                                          segments)

#             # Check JXT cost limit
#             if all(jc <= j_limit for jc in tmp_cost):
#                 # If valid, record the parent, the cost, and the jxt array
#                 tentative_cost = candidate.cost + d_candidate
#                 valid_parents.append((candidate, tentative_cost, tmp_cost))

#         if not valid_parents:
#             # No valid parent that satisfies j_limit
#             continue

#         # Among valid parents, pick the one with the minimum path cost
#         valid_parents.sort(key=lambda x: x[1])
#         best_parent, best_cost, best_jxt = valid_parents[0]

#         # Now finalize new_node with that best parent
#         new_node.parent = best_parent
#         new_node.cost = best_cost
#         d_parent = distance(best_parent, new_node)
#         new_node.time = best_parent.time + (d_parent / speed)
#         new_node.jxt_cost = best_jxt

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
#                     # Rewire
#                     near_node.parent = new_node
#                     near_node.cost = potential_cost
#                     near_node.time = tmp_rewire_node.time
#                     near_node.jxt_cost = tmp_rewire_jxt
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
#                      threshold=5.0,
#                      segments=None,
#                      gamma=0.9,
#                      dt=0.1,
#                      j_limit=180.0):
#     """
#     Extracts a path from the start to a node near the goal (within threshold),
#     then tries to add the exact goal if it doesn't break the JXT limit.

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
#             goal_node.jxt_cost = goal_node_jxt
#             path.append(goal_node)
#         else:
#             # If the direct step to the exact goal breaks j_limit,
#             # we just return the path up to 'best_node'
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
#     plt.title("RRT* Path Planning with JXT Constraint")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # ------------------------------
# # MAIN
# # ------------------------------
# def main():
#     # search_area = (xmin, xmax, ymin, ymax)
#     search_area = (0, 100, 0, 100)

#     start = Node(5, 5)
#     goal = Node(90, 90)

#     obstacles = []
#     obstacles.append((40, 40, 60, 60))
#     obstacles.append((20, 70, 30, 80))
#     obstacles.append((70, 20, 80, 30))

#     max_iter = 2500
#     step_size = 5
#     goal_sample_rate = 0.1
#     neighbor_radius = 10
#     threshold = 5.0
#     j_limit = 179.0  # Example j_limit value

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

#         print("\nFinal Path (Node, Time, pxq Values, jxt_cost):")
#         for node in final_path:
#             print(f"Node: ({node.x:.2f}, {node.y:.2f}), Time: {node.time:.4f}")
#             if node.pxq_values is not None:
#                 print("  pxq values:",
#                       [f"{val:.4f}" for val in node.pxq_values])
#             if node.jxt_cost is not None:
#                 print("  jxt_cost:", [f"{val:.4f}" for val in node.jxt_cost])

#         total_jxt = calculate_total_jxt_for_path(
#             final_path, get_obstacle_segments(obstacles), gamma=0.9, dt=0.1)
#         print("\nTotal jxt cost for the final path:")
#         if total_jxt:
#             print([f"{val:.4f}" for val in total_jxt])
#         else:
#             print("No path or single-node path (nothing to compute).")

#     else:
#         print("\nNo path found that satisfies j_limit!")

#     # Visualize
#     plot_rrt(nodes, final_path, obstacles, start, goal, search_area)

# if __name__ == "__main__":
#     main()

#THE CODE THAT BREAKS LIMIT AT THE LAST NODE

# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import random

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

# # ------------------------------
# # HELPER FUNCTIONS
# # ------------------------------
# def distance(n1, n2):
#     return math.hypot(n1.x - n2.x, n1.y - n2.y)

# def point_to_segment_distance(point, seg_start, seg_end):
#     """
#     Computes the minimum distance from a point to a line segment.
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
#         return math.hypot(apx, apy)
#     t = (apx * abx + apy * aby) / ab_squared
#     t = max(0, min(1, t))
#     closest_x = ax + t * abx
#     closest_y = ay + t * aby
#     return math.hypot(px - closest_x, py - closest_y)

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
#         if val == 0:
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
#         return sum(((gaussian[i] + gaussian[i - 1]) / 2) * dt
#                    for i in range(1, len(gaussian)))

# def calc_new_jxt_cost(from_node, to_node, gamma=0.9, dt=0.1, segments=None):
#     """
#     Computes the new jxt_cost for to_node given its parent (from_node)
#     using the recursive formula:
#       J(q,t₂) = γ^(t₂-t₁) * J(q,t₁) + ∫ₜ₁ᵗ₂ γ^(t₂ - τ) p(q, x(τ)) dτ.
#     This is done for each obstacle segment.

#     We discretize the time interval [from_node.time, to_node.time] with step dt.
#     """
#     deltaX = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
#     deltaT = to_node.time - from_node.time
#     if from_node.jxt_cost is None:
#         parent_jxt = [0.0 for _ in range(len(segments))]
#     else:
#         parent_jxt = from_node.jxt_cost
#     first_term = [(gamma**deltaT) * cost for cost in parent_jxt]
#     if deltaT == 0:
#         return first_term
#     n_steps = max(1, int((deltaX) / dt))
#     tau_values = [from_node.time + i * dt for i in range(n_steps)]
#     if tau_values[-1] < to_node.time:
#         tau_values.append(to_node.time)
#     f_values_per_seg = [[] for _ in range(len(segments))]
#     t1 = from_node.time
#     t2 = to_node.time
#     for tau in tau_values:
#         alpha = (tau - t1) / (deltaX) if deltaX > 0 else 0
#         x_tau = from_node.x + alpha * (to_node.x - from_node.x)
#         y_tau = from_node.y + alpha * (to_node.y - from_node.y)
#         temp_node = Node(x_tau, y_tau)
#         temp_pxq = compute_pxq_values(temp_node, segments)
#         weight = gamma**(t2 - tau)
#         for j in range(len(segments)):
#             f_values_per_seg[j].append(temp_pxq[j] * weight)
#     integral_term = []
#     for j in range(len(segments)):
#         integ = J_x_t_funct(f_values_per_seg[j], dt)
#         integral_term.append(integ)
#     new_jxt = [first_term[j] + integral_term[j] for j in range(len(segments))]
#     return new_jxt

# # ------------------------------
# # STEERING AND COLLISION
# # ------------------------------
# def steer(from_node, to_point, step_size, segments):
#     """
#     Steers from from_node toward to_point by at most step_size.
#     Computes the new node’s cost, time, and its pxq_values.
#     """
#     speed = 0.1
#     dx = to_point[0] - from_node.x
#     dy = to_point[1] - from_node.y
#     dist = math.hypot(dx, dy)
#     if dist < step_size:
#         new_x, new_y = to_point[0], to_point[1]
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
#     new_node.pxq_values = compute_pxq_values(new_node, segments)
#     return new_node

# def is_inside_box(point, box):
#     """
#     Checks if point (x,y) is inside the given box (xmin, ymin, xmax, ymax).
#     """
#     return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

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
#     Returns the node in nodes that is nearest to the given point.
#     """
#     nearest = nodes[0]
#     min_dist = math.hypot(nearest.x - point[0], nearest.y - point[1])
#     for node in nodes:
#         d = math.hypot(node.x - point[0], node.y - point[1])
#         if d < min_dist:
#             nearest = node
#             min_dist = d
#     return nearest

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
#              j_limit=7.0):
#     """
#     Runs the RRT* algorithm with JXT cost constraint.
#     Returns:
#       - nodes: all nodes in the tree.
#       - path_progress: list of tuples (best_path_length, number_of_nodes)
#     """
#     segments = get_obstacle_segments(obstacles)
#     nodes = [start]
#     start.pxq_values = compute_pxq_values(start, segments)
#     start.jxt_cost = [0.0 for _ in range(len(segments))]
#     best_path_length = float('inf')
#     path_progress = []
#     speed = 0.1

#     for i in range(max_iter):
#         if random.random() < goal_sample_rate:
#             sample = (goal.x, goal.y)
#         else:
#             sample = (random.uniform(search_area[0], search_area[1]),
#                       random.uniform(search_area[2], search_area[3]))
#         nearest_node = get_nearest_node(nodes, sample)
#         new_node = steer(nearest_node, sample, step_size, segments)

#         if not collision_check(nearest_node, new_node, obstacles):
#             print(f"Iter: {i}, collision detected, skipping.")
#             continue

#         near_nodes = []
#         for node in nodes:
#             if math.hypot(node.x - new_node.x,
#                           node.y - new_node.y) <= neighbor_radius:
#                 if collision_check(node, new_node, obstacles):
#                     near_nodes.append(node)
#         # Include the nearest_node in parent candidates
#         parent_candidates = near_nodes.copy()
#         if nearest_node not in parent_candidates:
#             parent_candidates.append(nearest_node)
#         valid_parents = []
#         for candidate in parent_candidates:
#             # Compute tentative jxt_cost for new_node if connected to candidate
#             tentative_jxt = calc_new_jxt_cost(candidate, new_node, gamma, dt,
#                                               segments)
#             # Check if all jxt values are <= j_limit
#             if all(jxt <= j_limit for jxt in tentative_jxt):
#                 d_candidate = distance(candidate, new_node)
#                 tentative_cost = candidate.cost + d_candidate
#                 valid_parents.append(
#                     (candidate, tentative_cost, tentative_jxt))
#         if valid_parents:
#             # Find the parent with the minimum tentative_cost
#             valid_parents.sort(key=lambda x: x[1])
#             best_parent, best_cost, best_jxt = valid_parents[0]
#             new_node.parent = best_parent
#             new_node.cost = best_cost
#             d_parent = distance(best_parent, new_node)
#             new_node.time = best_parent.time + d_parent / speed
#             new_node.jxt_cost = best_jxt
#             nodes.append(new_node)
#             # Proceed with rewiring...
#         else:
#             print(f"Iter: {i}, new node exceeds j_limit, skipping.")
#             continue

#         # Rewiring
#         for near_node in near_nodes:
#             d_rewire = distance(new_node, near_node)
#             potential_cost = new_node.cost + d_rewire
#             if potential_cost < near_node.cost:
#                 # Compute tentative jxt_cost for near_node if parent is new_node
#                 tentative_jxt = calc_new_jxt_cost(new_node, near_node, gamma,
#                                                   dt, segments)
#                 if all(jxt <= j_limit for jxt in tentative_jxt):
#                     if collision_check(new_node, near_node, obstacles):
#                         near_node.parent = new_node
#                         near_node.cost = potential_cost
#                         near_node.time = new_node.time + d_rewire / speed
#                         near_node.jxt_cost = tentative_jxt

#         print(f"Iter: {i}, number of nodes: {len(nodes)}")

#         current_path = get_path_to_goal(nodes, goal, threshold, segments)
#         if current_path is not None:
#             current_length = compute_path_length([(n.x, n.y)
#                                                   for n in current_path])
#             if current_length < best_path_length:
#                 best_path_length = current_length
#                 path_progress.append((best_path_length, len(nodes)))
#     return nodes, path_progress

# def get_path_to_goal(nodes, goal, threshold=1.0, segments=None):
#     """
#     Extracts a path from the start to goal (list of Node objects).
#     If the last node is not exactly at the goal, an extra node is appended.
#     Uses the provided segments to compute the final node's pxq and jxt_cost.
#     """
#     goal_nodes = []
#     for node in nodes:
#         if math.hypot(node.x - goal.x, node.y - goal.y) <= threshold:
#             goal_nodes.append(node)
#     if not goal_nodes:
#         return None
#     best_node = min(goal_nodes, key=lambda n: n.cost)
#     path = []
#     node = best_node
#     while node is not None:
#         path.append(node)
#         node = node.parent
#     path.reverse()
#     last = path[-1]
#     if math.hypot(last.x - goal.x, last.y - goal.y) > 1e-6:
#         d = math.hypot(goal.x - last.x, goal.y - last.y)
#         speed = 0.1
#         goal_time = last.time + d / speed
#         goal_node = Node(goal.x, goal.y)
#         goal_node.time = goal_time
#         goal_node.parent = last
#         if segments is not None:
#             goal_node.pxq_values = compute_pxq_values(goal_node, segments)
#             goal_node.jxt_cost = calc_new_jxt_cost(last, goal_node, 0.9, 0.1,
#                                                    segments)
#         else:
#             goal_node.pxq_values = []
#             goal_node.jxt_cost = []
#         path.append(goal_node)
#     return path

# def compute_path_length(path):
#     """
#     Given a list of (x,y) coordinates, computes total path length.
#     """
#     path_length = 0.0
#     for i in range(1, len(path)):
#         path_length += math.hypot(path[i][0] - path[i - 1][0],
#                                   path[i][1] - path[i - 1][1])
#     return path_length

# # ------------------------------
# # NEW FUNCTION: CALCULATE TOTAL JXT COST FOR PATH
# # ------------------------------
# def calculate_total_jxt_for_path(path, segments, gamma=0.9, dt=0.1):
#     """
#     Given a path (list of Node objects), this function computes the cumulative
#     jxt cost along the path by using calc_new_jxt_cost from one node to the next.
#     It returns the cumulative jxt cost (a list, one per obstacle segment) for the final node.
#     """
#     if len(path) < 2:
#         return None
#     cumulative_jxt = calc_new_jxt_cost(path[0], path[1], gamma, dt, segments)
#     for i in range(2, len(path)):
#         dummy_parent = Node(path[i - 1].x, path[i - 1].y)
#         dummy_parent.time = path[i - 1].time
#         dummy_parent.jxt_cost = cumulative_jxt
#         cumulative_jxt = calc_new_jxt_cost(dummy_parent, path[i], gamma, dt,
#                                            segments)
#     return cumulative_jxt

# # ------------------------------
# # PLOTTING
# # ------------------------------
# def plot_rrt(nodes, path, obstacles, start, goal, search_area):
#     plt.figure(figsize=(8, 8))
#     for node in nodes:
#         if node.parent is not None:
#             plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")
#     if path is not None:
#         px = [p.x for p in path]
#         py = [p.y for p in path]
#         plt.plot(px, py, "-r", linewidth=2, label="Final path")
#     for obs in obstacles:
#         rect = plt.Rectangle((obs[0], obs[1]),
#                              obs[2] - obs[0],
#                              obs[3] - obs[1],
#                              color="gray",
#                              alpha=0.7)
#         plt.gca().add_patch(rect)
#     plt.plot(start.x, start.y, "bs", markersize=10, label="Start")
#     plt.plot(goal.x, goal.y, "ms", markersize=10, label="Goal")
#     plt.xlim(search_area[0], search_area[1])
#     plt.ylim(search_area[2], search_area[3])
#     plt.title("RRT* Path Planning with JXT Constraint")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # ------------------------------
# # MAIN
# # ------------------------------
# def main():
#     # search_area = (0, 100, 0, 100)
#     # start = Node(5, 5)
#     # goal = Node(90, 90)
#     search_area = (-5, 30, -5, 20)
#     start = Node(5.5, 1.0)
#     goal = Node(22.5, 10)
#     obstacles = []
#     # obstacles.append((40, 40, 60, 60))
#     # obstacles.append((20, 70, 30, 80))
#     # obstacles.append((70, 20, 80, 30))
#     obstacles.append((7.4, 2.3, 9.3, 3.8))
#     obstacles.append((10.1, 2.3, 10.9, 4.5))  # 720 Albany
#     obstacles.append((11.8, 2.5, 13.9, 4.9))  # 710 Albany
#     obstacles.append((14.5, 3.7, 17.0, 4.9))  # 700 Albany
#     obstacles.append((14.6, 2.1, 17.1, 3.3))  # 670 Albany
#     obstacles.append((17.8, 4.0, 20.1, 5.2))  # 650 Albany
#     obstacles.append((20.7, 2.3, 23.3, 3.6))  # 620 Albany
#     obstacles.append((23.5, 0.0, 26.1, 1.9))  # 610 Albany
#     obstacles.append((21.0, 6.2, 21.7, 7.1))  # 615 Albany
#     obstacles.append((22.4, 6.2, 23.1, 7.1))  # 609 Albany
#     obstacles.append((24.9, 7.6, 25.8, 9.2))  # 100 E.Canton
#     obstacles.append((24.9, 9.6, 26.0, 11.2))  # Employee Parking Lot
#     obstacles.append((24.9, 13.6, 25.8, 14.5))  # 660 Harrison
#     obstacles.append((19.9, 11.8, 21.6, 14.1))  # BMC DOCTORS
#     obstacles.append((14.7, 11.4, 15.8, 12.8))  # VOSE
#     obstacles.append((9.6, 10.8, 10.2, 12.3))  # BCD
#     obstacles.append((7.1, 10.8, 7.7, 12.3))  # FGH
#     obstacles.append((0.0, 6.1, 2.2, 12.2))  # NORTHHAMPTON SQUARE
#     obstacles.append((15.6, 6.2, 17.0, 9.8))  # SOLOMON CARTER
#     obstacles.append((17.6, 8.3, 20.8, 11.6))  # 88E NEWTON
#     obstacles.append((17.6, 6.3, 19.7, 8.1))  # GOLDMAN SCHOOL OF DENTAL
#     obstacles.append((11.9, 6.0, 14.0, 8.3))  # SCHOOL OF PUBLIC HEALTH
#     obstacles.append((2.2, 8.4, 5.1, 10.3))  # BMC YAWKEY CENTER
#     obstacles.append((5.1, 6.4, 7.8, 9.9))  # BMC MENINO
#     obstacles.append((7.8, 8.0, 10.8, 9.7))  # BMC MOAKLEY
#     # obstacles.append((-1.0, 5.474, -0.385, 6.168))  # Speaker Obstacle
#     # obstacles.append((0.25, 5.455, 0.57, 5.825))  # Box 1
#     # obstacles.append((1.41, 5.455, 1.73, 5.825))  # Box 2

#     max_iter = 10500
#     #step_size = 5
#     step_size = 1
#     goal_sample_rate = 0.2
#     #neighbor_radius = 10
#     neighbor_radius = 2
#     threshold = 1.0
#     j_limit = 7.0  # Example j_limit value

#     segments = get_obstacle_segments(obstacles)
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
#     final_path = get_path_to_goal(nodes, goal, threshold, segments)
#     print("\nPath Length Samples:")
#     print(path_progress)
#     if final_path is not None:
#         # Verify all nodes in the path satisfy j_limit
#         all_within_limit = True
#         for node in final_path:
#             if any(jxt > j_limit for jxt in node.jxt_cost):
#                 all_within_limit = False
#                 break
#         if all_within_limit:
#             print(
#                 "\nAll nodes in the final path satisfy the j_limit constraint."
#             )
#         else:
#             print("\nWARNING: Some nodes in the path exceed the j_limit!")
#         print("\nFinal Path with Node Times, pxq Values, and jxt_cost:")
#         for node in final_path:
#             print(f"Node: ({node.x:.2f}, {node.y:.2f}), Time: {node.time:.4f}")
#             if node.pxq_values is not None:
#                 print("  pxq values:",
#                       [f"{val:.4f}" for val in node.pxq_values])
#             if node.jxt_cost is not None:
#                 print("  jxt_cost:", [f"{val:.4f}" for val in node.jxt_cost])
#         total_jxt = calculate_total_jxt_for_path(final_path,
#                                                  segments,
#                                                  gamma=0.9,
#                                                  dt=0.1)
#         print("\nTotal jxt cost for the final path:")
#         print([f"{val:.4f}" for val in total_jxt])
#     plot_rrt(nodes, final_path, obstacles, start, goal, search_area)

# if __name__ == "__main__":
#     main()

# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import random

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

# # ------------------------------
# # HELPER FUNCTIONS
# # ------------------------------
# def distance(n1, n2):
#     return math.hypot(n1.x - n2.x, n1.y - n2.y)

# def point_to_segment_distance(point, seg_start, seg_end):
#     """
#     Computes the minimum distance from a point to a line segment.
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
#         return math.hypot(apx, apy)
#     t = (apx * abx + apy * aby) / ab_squared
#     t = max(0, min(1, t))
#     closest_x = ax + t * abx
#     closest_y = ay + t * aby
#     return math.hypot(px - closest_x, py - closest_y)

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
#         if val == 0:
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
#         return sum(((gaussian[i] + gaussian[i - 1]) / 2) * dt
#                    for i in range(1, len(gaussian)))

# def calc_new_jxt_cost(from_node, to_node, gamma=0.9, dt=0.1, segments=None):
#     """
#     Computes the new jxt_cost for to_node given its parent (from_node)
#     using the recursive formula:
#       J(q,t₂) = γ^(t₂-t₁) * J(q,t₁) + ∫ₜ₁ᵗ₂ γ^(t₂ - τ) p(q, x(τ)) dτ.
#     This is done for each obstacle segment.

#     We discretize the time interval [from_node.time, to_node.time] with step dt.
#     """
#     deltaT = to_node.time - from_node.time
#     if from_node.jxt_cost is None:
#         parent_jxt = [0.0 for _ in range(len(segments))]
#     else:
#         parent_jxt = from_node.jxt_cost
#     first_term = [(gamma**deltaT) * cost for cost in parent_jxt]
#     if deltaT == 0:
#         return first_term
#     n_steps = max(1, int(deltaT / dt))
#     tau_values = [from_node.time + i * dt for i in range(n_steps)]
#     if tau_values[-1] < to_node.time:
#         tau_values.append(to_node.time)
#     f_values_per_seg = [[] for _ in range(len(segments))]
#     t1 = from_node.time
#     t2 = to_node.time
#     for tau in tau_values:
#         alpha = (tau - t1) / deltaT if deltaT > 0 else 0
#         x_tau = from_node.x + alpha * (to_node.x - from_node.x)
#         y_tau = from_node.y + alpha * (to_node.y - from_node.y)
#         temp_node = Node(x_tau, y_tau)
#         temp_pxq = compute_pxq_values(temp_node, segments)
#         weight = gamma**(t2 - tau)
#         for j in range(len(segments)):
#             f_values_per_seg[j].append(temp_pxq[j] * weight)
#     integral_term = []
#     for j in range(len(segments)):
#         integ = J_x_t_funct(f_values_per_seg[j], dt)
#         integral_term.append(integ)
#     new_jxt = [first_term[j] + integral_term[j] for j in range(len(segments))]
#     return new_jxt

# # ------------------------------
# # STEERING AND COLLISION
# # ------------------------------
# def steer(from_node, to_point, step_size, segments):
#     """
#     Steers from from_node toward to_point by at most step_size.
#     Computes the new node’s cost, time, and its pxq_values.
#     """
#     speed = 0.1
#     dx = to_point[0] - from_node.x
#     dy = to_point[1] - from_node.y
#     dist = math.hypot(dx, dy)
#     if dist < step_size:
#         new_x, new_y = to_point[0], to_point[1]
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
#     new_node.pxq_values = compute_pxq_values(new_node, segments)
#     return new_node

# def is_inside_box(point, box):
#     """
#     Checks if point (x,y) is inside the given box (xmin, ymin, xmax, ymax).
#     """
#     return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

# def collision_check(node1, node2, obstacles):
#     """
#     Checks if the straight-line path between node1 and node2 is collision-free.
#     """
#     resolution = 0.5
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
#     Returns the node in nodes that is nearest to the given point.
#     """
#     nearest = nodes[0]
#     min_dist = math.hypot(nearest.x - point[0], nearest.y - point[1])
#     for node in nodes:
#         d = math.hypot(node.x - point[0], node.y - point[1])
#         if d < min_dist:
#             nearest = node
#             min_dist = d
#     return nearest

# # ------------------------------
# # RRT* ALGORITHM
# # ------------------------------
# def rrt_star(start,
#              goal,
#              obstacles,
#              search_area,
#              max_iter=500,
#              step_size=5,
#              goal_sample_rate=0.1,
#              neighbor_radius=10,
#              threshold=5.0,
#              gamma=0.9,
#              dt=0.1):
#     """
#     Runs the RRT* algorithm.
#     Returns:
#       - nodes: all nodes in the tree.
#       - path_progress: list of tuples (best_path_length, number_of_nodes)
#     """
#     segments = get_obstacle_segments(obstacles)
#     nodes = [start]
#     start.pxq_values = compute_pxq_values(start, segments)
#     start.jxt_cost = [0.0 for _ in range(len(segments))]
#     best_path_length = float('inf')
#     path_progress = []
#     speed = 0.1

#     for i in range(max_iter):
#         if random.random() < goal_sample_rate:
#             sample = (goal.x, goal.y)
#         else:
#             sample = (random.uniform(search_area[0], search_area[1]),
#                       random.uniform(search_area[2], search_area[3]))
#         nearest_node = get_nearest_node(nodes, sample)
#         new_node = steer(nearest_node, sample, step_size, segments)

#         if not collision_check(nearest_node, new_node, obstacles):
#             print(f"Iter: {i}, number of nodes: {len(nodes)}")
#             continue

#         near_nodes = []
#         for node in nodes:
#             if math.hypot(node.x - new_node.x,
#                           node.y - new_node.y) <= neighbor_radius:
#                 if collision_check(node, new_node, obstacles):
#                     near_nodes.append(node)
#         if near_nodes:
#             costs = [
#                 node.cost +
#                 math.hypot(node.x - new_node.x, node.y - new_node.y)
#                 for node in near_nodes
#             ]
#             min_cost = min(costs)
#             min_index = costs.index(min_cost)
#             new_node.parent = near_nodes[min_index]
#             new_node.cost = min_cost
#             d = math.hypot(near_nodes[min_index].x - new_node.x,
#                            near_nodes[min_index].y - new_node.y)
#             new_node.time = near_nodes[min_index].time + d / speed
#             new_node.jxt_cost = calc_new_jxt_cost(near_nodes[min_index],
#                                                   new_node, gamma, dt,
#                                                   segments)
#         else:
#             new_node.jxt_cost = calc_new_jxt_cost(new_node.parent, new_node,
#                                                   gamma, dt, segments)

#         nodes.append(new_node)

#         for near_node in near_nodes:
#             d_rewire = math.hypot(near_node.x - new_node.x,
#                                   near_node.y - new_node.y)
#             potential_cost = new_node.cost + d_rewire
#             potential_time = new_node.time + d_rewire / speed
#             if potential_cost < near_node.cost:
#                 if collision_check(new_node, near_node, obstacles):
#                     near_node.parent = new_node
#                     near_node.cost = potential_cost
#                     near_node.time = potential_time
#                     near_node.jxt_cost = calc_new_jxt_cost(
#                         new_node, near_node, gamma, dt, segments)

#         print(f"Iter: {i}, number of nodes: {len(nodes)}")

#         current_path = get_path_to_goal(nodes, goal, threshold, segments)
#         if current_path is not None:
#             current_length = compute_path_length([(n.x, n.y)
#                                                   for n in current_path])
#             if current_length < best_path_length:
#                 best_path_length = current_length
#                 path_progress.append((best_path_length, len(nodes)))
#     return nodes, path_progress

# def get_path_to_goal(nodes, goal, threshold=5.0, segments=None):
#     """
#     Extracts a path from the start to goal (list of Node objects).
#     If the last node is not exactly at the goal, an extra node is appended.
#     Uses the provided segments to compute the final node's pxq and jxt_cost.
#     """
#     goal_nodes = []
#     for node in nodes:
#         if math.hypot(node.x - goal.x, node.y - goal.y) <= threshold:
#             goal_nodes.append(node)
#     if not goal_nodes:
#         return None
#     best_node = min(goal_nodes, key=lambda n: n.cost)
#     path = []
#     node = best_node
#     while node is not None:
#         path.append(node)
#         node = node.parent
#     path.reverse()
#     last = path[-1]
#     if math.hypot(last.x - goal.x, last.y - goal.y) > 1e-6:
#         d = math.hypot(goal.x - last.x, goal.y - last.y)
#         speed = 0.1
#         goal_time = last.time + d / speed
#         goal_node = Node(goal.x, goal.y)
#         goal_node.time = goal_time
#         goal_node.parent = last
#         if segments is not None:
#             goal_node.pxq_values = compute_pxq_values(goal_node, segments)
#             goal_node.jxt_cost = calc_new_jxt_cost(last, goal_node, 0.9, 0.1,
#                                                    segments)
#         else:
#             goal_node.pxq_values = []
#             goal_node.jxt_cost = []
#         path.append(goal_node)
#     return path

# def compute_path_length(path):
#     """
#     Given a list of (x,y) coordinates, computes total path length.
#     """
#     path_length = 0.0
#     for i in range(1, len(path)):
#         path_length += math.hypot(path[i][0] - path[i - 1][0],
#                                   path[i][1] - path[i - 1][1])
#     return path_length

# # ------------------------------
# # NEW FUNCTION: CALCULATE TOTAL JXT COST FOR PATH
# # ------------------------------
# def calculate_total_jxt_for_path(path, segments, gamma=0.9, dt=0.1):
#     """
#     Given a path (list of Node objects), this function computes the cumulative
#     jxt cost along the path by using calc_new_jxt_cost from one node to the next.
#     It returns the cumulative jxt cost (a list, one per obstacle segment) for the final node.
#     """
#     if len(path) < 2:
#         return None
#     cumulative_jxt = calc_new_jxt_cost(path[0], path[1], gamma, dt, segments)
#     for i in range(2, len(path)):
#         dummy_parent = Node(path[i - 1].x, path[i - 1].y)
#         dummy_parent.time = path[i - 1].time
#         dummy_parent.jxt_cost = cumulative_jxt
#         cumulative_jxt = calc_new_jxt_cost(dummy_parent, path[i], gamma, dt,
#                                            segments)
#     return cumulative_jxt

# # ------------------------------
# # PLOTTING
# # ------------------------------
# def plot_rrt(nodes, path, obstacles, start, goal, search_area):
#     plt.figure(figsize=(8, 8))
#     for node in nodes:
#         if node.parent is not None:
#             plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")
#     if path is not None:
#         px = [p.x for p in path]
#         py = [p.y for p in path]
#         plt.plot(px, py, "-r", linewidth=2, label="Final path")
#     for obs in obstacles:
#         rect = plt.Rectangle((obs[0], obs[1]),
#                              obs[2] - obs[0],
#                              obs[3] - obs[1],
#                              color="gray",
#                              alpha=0.7)
#         plt.gca().add_patch(rect)
#     plt.plot(start.x, start.y, "bs", markersize=10, label="Start")
#     plt.plot(goal.x, goal.y, "ms", markersize=10, label="Goal")
#     plt.xlim(search_area[0], search_area[1])
#     plt.ylim(search_area[2], search_area[3])
#     plt.title("RRT* Path Planning")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # ------------------------------
# # MAIN
# # ------------------------------
# def main():
#     search_area = (0, 100, 0, 100)
#     start = Node(5, 5)
#     goal = Node(90, 90)
#     obstacles = []
#     obstacles.append((40, 40, 60, 60))
#     obstacles.append((20, 70, 30, 80))
#     obstacles.append((70, 20, 80, 30))
#     max_iter = 2500
#     step_size = 5
#     goal_sample_rate = 0.1
#     neighbor_radius = 10
#     threshold = 5.0

#     segments = get_obstacle_segments(obstacles)
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
#                                     dt=0.1)
#     final_path = get_path_to_goal(nodes, goal, threshold, segments)
#     print("\nPath Length Samples:")
#     print(path_progress)
#     if final_path is not None:
#         print("\nFinal Path with Node Times, pxq Values, and jxt_cost:")
#         for node in final_path:
#             print(f"Node: ({node.x:.2f}, {node.y:.2f}), Time: {node.time:.4f}")
#             if node.pxq_values is not None:
#                 print("  pxq values:",
#                       [f"{val:.4f}" for val in node.pxq_values])
#             if node.jxt_cost is not None:
#                 print("  jxt_cost:", [f"{val:.4f}" for val in node.jxt_cost])
#         total_jxt = calculate_total_jxt_for_path(final_path,
#                                                  segments,
#                                                  gamma=0.9,
#                                                  dt=0.1)
#         print("\nTotal jxt cost for the final path:")
#         print([f"{val:.4f}" for val in total_jxt])
#     plot_rrt(nodes, final_path, obstacles, start, goal, search_area)

# if __name__ == "__main__":
#     main()
