import numpy as np
from scipy.spatial import ConvexHull, KDTree, Delaunay
from scipy.optimize import minimize

# Define the points forming the convex hull

class ProjectionHandler:
    
    def __init__(self, points):
        self.points = points.copy()
        self.hull = ConvexHull(points)
        self.tree = KDTree(points)
        self.tri = Delaunay(points)

    def project_points(self, query_points):
        
        # Check which points lie inside the convex hull
        inside_hull = self.tri.find_simplex(query_points) >= 0
        projections = query_points.copy()  # Initialize with the input points

        if np.all(inside_hull):
            return projections  # If all points are inside, return them directly

        # For points outside the hull, calculate projections
        outside_points = query_points[~inside_hull]

        # Find the nearest vertices for all outside points using KDTree
        _, nearest_indices = self.tree.query(outside_points)
        nearest_vertices = self.points[nearest_indices]

        # Define constraints for all points
        constraints = []
        for simplex in self.hull.simplices:
            vertices = self.points[simplex]
            A = vertices[1] - vertices[0]
            b = vertices[0]
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, A=A, b=b: np.dot(A, x - b)
            })

        # Minimize distances for all outside points
        results = []
        for i, query_point in enumerate(outside_points):
            result = minimize(
                lambda x, y=query_point: np.sum((x - y)**2),
                nearest_vertices[i],
                constraints=constraints,
                method='SLSQP'
            )
            results.append(result.x)
        
        # Update projections for points outside the hull
        projections[~inside_hull] = np.array(results)

        return projections
    
if __name__ == "__main__":
    
    import sys 
    import time

    num_hull_points, num_query_points = int(sys.argv[1]), int(sys.argv[2])
    original_points = np.random.rand(num_hull_points, 3)
    query_points = np.random.rand(num_query_points, 3)

    start = time.time()
    projection_handler = ProjectionHandler(original_points)
    projections = projection_handler.project_points(query_points)
    end = time.time()

    print(projection_handler.hull.vertices.shape)
    print(end - start)
    