import numpy as np
from projection import ProjectionHandler

def get_mean_value_coordinates(points, vertices, triangles, epsilon=1e-8):
    """
    Vectorized mean value coordinates calculation for mesh interpolation.

    Parameters:
    - points: np.ndarray, shape (m, 3), query points
    - vertices: np.ndarray, shape (n, 3), mesh vertices
    - triangles: np.ndarray, shape (k, 3), optional triangle indices
    - epsilon: float, numerical stability threshold

    Returns:
    - results: np.ndarray, interpolated values for each point
    """

    # Precompute vertex values (identity matrix)
    values = np.eye(len(vertices))

    # Compute distances from points to vertices
    projections = ProjectionHandler(vertices)
    points = projections.project_points(points)
    point_vertex_distances = np.linalg.norm(
        points[:, np.newaxis, :] - vertices[np.newaxis, :, :], 
        axis=2
    )

    # Find closest vertices
    closest_vertices = np.argmin(point_vertex_distances, axis=1)
    close_enough = point_vertex_distances[np.arange(len(points)), closest_vertices] < epsilon
    direct_results = values[closest_vertices]
    direct_results[~close_enough] = None

    # Mask for points needing complex computation
    compute_mask = ~close_enough

    # Preallocate results
    results = np.full(len(points), None, dtype=object)
    results[close_enough] = direct_results[close_enough]

    # Process remaining points
    for i in np.where(compute_mask)[0]:
        x = points[i]
        point_result = None

        for triangle in triangles:
            p1, p2, p3 = vertices[triangle]
            f1, f2, f3 = values[triangle]

            # Normalized vectors
            u1 = (p1 - x) / np.linalg.norm(p1 - x)
            u2 = (p2 - x) / np.linalg.norm(p2 - x)
            u3 = (p3 - x) / np.linalg.norm(p3 - x)

            # Similar computations as original method
            l1 = np.linalg.norm(u2 - u3)
            l2 = np.linalg.norm(u3 - u1)
            l3 = np.linalg.norm(u1 - u2)

            theta1 = 2 * np.arcsin(l1 / 2)
            theta2 = 2 * np.arcsin(l2 / 2)
            theta3 = 2 * np.arcsin(l3 / 2)

            h = (theta1 + theta2 + theta3) / 2

            if np.pi - h < epsilon:
                # Barycentric coordinates
                w1 = np.sin(theta2) * l1 / l2
                w2 = np.sin(theta3) * l2 / l3
                w3 = np.sin(theta1) * l3 / l1
                weights = np.array([w1, w2, w3])
                point_result = np.dot(weights / weights.sum(), [f1, f2, f3])
                break

        results[i] = point_result

    return np.array(results)

def get_weights(image, hvertices, hfaces):
    N, M, _ = image.shape
    points = image.reshape((N*M, 3))
    weights = get_mean_value_coordinates(points, hvertices, hfaces)
    weights = weights.reshape((N, M, hvertices.shape[0]))
    return weights

if __name__ == "__main__":
    pass