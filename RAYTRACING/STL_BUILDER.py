import numpy as np
from stl import mesh
import os


def get_box_vertices(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return np.array([[x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
                     [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]])


def create_box_mesh(vertices):
    faces = np.array([
        [0, 3, 1],
        [1, 3, 2],  # Bottom
        [0, 1, 5],
        [0, 5, 4],  # Front
        [1, 2, 6],
        [1, 6, 5],  # Right
        [2, 3, 7],
        [2, 7, 6],  # Back
        [3, 0, 4],
        [3, 4, 7],  # Left
        [4, 5, 6],
        [4, 6, 7]  # Top
    ])
    m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = vertices[f[j], :]
    return m


# Define objects
objects = {
    #"room": [(-1.000, -0.700, -0.030), (3.000, 8.300, 3.000)],
    "room": [(-1.000, -0.700, -0.030), (5.000, 8.300, 4.000)],
    #"box1": [(-1.000, 5.508, -0.030), (-0.606, 6.199, 1.254)],
    #"box2": [(0.257, 5.508, -0.030), (0.580, 6.199, 1.254)],
    #"box3": [(1.520, 5.508, -0.030), (1.830, 6.199, 1.254)],
}

# Create and combine meshes
meshes = []
for p1, p2 in objects.values():
    v = get_box_vertices(p1, p2)
    meshes.append(create_box_mesh(v))

combined = mesh.Mesh(np.concatenate([m.data for m in meshes]))

# Save path
output_path = '/Users/idrisseidu/Documents/MATLAB/upload/room1_only.stl'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
combined.save(output_path)
