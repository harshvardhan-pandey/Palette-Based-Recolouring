import numpy as np
from scipy.spatial import ConvexHull, KDTree, Delaunay
from scipy.optimize import minimize
from joblib import Parallel, delayed

class Projector:
    
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
                tol=10**(-9),
                method="SLSQP"
            )
        
        return result.x
    
    def project_points(self, query_points):

        inside_hull = self.tri.find_simplex(query_points) >= 0
        if np.all(inside_hull):
            return query_points.copy()
        
        outside_points = query_points[~inside_hull]
        outside_projections = np.array([
            self.project_point(point) for point in outside_points
        ])
        
        projections = query_points.copy()
        projections[~inside_hull] = outside_projections
        
        return projections
    
    def project_points_parallel(self, query_points, n_jobs=-1):

        
        inside_hull = self.tri.find_simplex(query_points) >= 0
        if np.all(inside_hull):
            return query_points.copy()
        
        outside_points = query_points[~inside_hull]
        _, nearest_indices = self.tree.query(outside_points)
        nearest_vertices = self.points[nearest_indices]
        
        def project_single_point(point, vertex):
            # Implement your specific projection logic here
            # This is a simplified example
            result = minimize(
                lambda x: np.sum((x - point)**2),
                x0=vertex,
                constraints=self.constraints,
                tol = 10**(-9),
                method='SLSQP'
            )
            return result.x
        
        # Parallel processing of projections
        outside_projections = Parallel(n_jobs=n_jobs)(
            delayed(project_single_point)(point, vertex) 
            for point, vertex in zip(outside_points, nearest_vertices)
        )
        
        projections = query_points.copy()
        projections[~inside_hull] = np.array(outside_projections)
        
        return projections

    
if __name__ == "__main__":
    
    points = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]) 
    ds = Projector(points)
    print(ds.hull)
    print(ds.tri.find_simplex([-1, -1, -1]))
    