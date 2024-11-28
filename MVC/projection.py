import numpy as np
from scipy.spatial import ConvexHull, KDTree, Delaunay
from scipy.optimize import minimize

class ProjectionHandler:
    
    def __init__(self, points):

        self.points = points.copy()
        self.hull = ConvexHull(points)
        self.tree = KDTree(points)
        self.tri = Delaunay(points)

        self.A = self.hull.equations[:, :-1]
        self.b = self.hull.equations[:, -1]
        self.constraints = [{'type': 'ineq', 'fun': lambda x: - self.b - np.dot(self.A, x)}]

    def project_point(self, query_point):
        
        if self.tri.find_simplex(query_point)>=0:
            return query_point.copy()
        
        _, nearest_idx = self.tree.query(query_point)

        result = minimize(
                lambda x, y=query_point: np.sum((x - y)**2),
                x0=self.points[nearest_idx],
                constraints=self.constraints,
                hess=lambda x: 2 * np.eye(x.shape[0]),
                tol = 1e-9,
                method="trust-constr"
            )
        
        return result.x

    def project_points(self, query_points, BLOCK_LENGTH = 30):
        
        inside_hull = self.tri.find_simplex(query_points) >= 0
        projections = query_points.copy()

        if np.all(inside_hull):
            return projections 
        
        outside_points = query_points[~inside_hull]
        outside_projections = outside_points.copy()
        _, nearest_indices = self.tree.query(outside_points)
        nearest_vertices = self.points[nearest_indices]

        N = nearest_vertices.shape[0]
        for block_start in range(0, N, BLOCK_LENGTH):

            block_end = min(block_start + BLOCK_LENGTH, N)
            actual_block_size = block_end - block_start
            outside_points_block = outside_points[block_start: block_end].flatten()
            nearest_vertices_block = nearest_vertices[block_start: block_end, :].flatten()

            A = np.kron(np.eye(actual_block_size), self.A)
            b = np.tile(self.b, actual_block_size)
            constraints = [{'type': 'ineq', 'fun': lambda x: - b - np.dot(A, x)}]

            result = minimize(
                lambda x: np.sum((x - outside_points_block)**2),
                x0=nearest_vertices_block,
                constraints=constraints,
                hess=lambda x: 2 * np.eye(x.shape[0]),
                tol = 1e-9,
                method="trust-constr"
            )

            block_result = result.x
            length = block_result.shape[0]
            block_result = block_result.reshape((actual_block_size, length/actual_block_size))
            outside_projections[block_start:block_end] = block_result

        projections[~inside_hull] = outside_projections
        return projections

    
if __name__ == "__main__":
    
    points = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]) 
    ds = ProjectionHandler(points)
    print(ds.hull)
    print(ds.tri.find_simplex([-1, -1, -1]))
    