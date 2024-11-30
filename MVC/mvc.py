import numpy as np
from projection import Projector

class MVC:

    def __init__(self, pallete_points):
        self.pallete_points = pallete_points.copy()
        self.f_values = np.eye(pallete_points.shape[0], dtype=np.float32)
        self.projector = Projector(self.pallete_points)
        self.triangles = self.projector.hull.simplices
        self.vertices = self.pallete_points[self.projector.hull.vertices]

    @staticmethod
    def safe_arcsin(val):
        return np.arcsin(np.clip(val, -1.0, 1.0))


    def compute_fx(self, x, epsilon=1e-6):  

        x = self.projector.project_point(x)
        distances = np.linalg.norm(self.vertices - x, axis=1)
        
        close_vertices = np.where(distances < epsilon)[0]
        if close_vertices.size > 0:
            return self.f_values[close_vertices[0]]

        # Compute unit vectors u_j
        u_vectors = (self.vertices - x) / distances[:, np.newaxis]  # Shape: (n_vertices, dim)

        totalF = np.zeros_like(self.f_values[0])
        totalW = 0.0

        for tri in self.triangles:
            idx1, idx2, idx3 = tri
            f1, f2, f3 = self.f_values[idx1], self.f_values[idx2], self.f_values[idx3]

            # Get unit vectors for the triangle vertices
            u1, u2, u3 = u_vectors[idx1], u_vectors[idx2], u_vectors[idx3]

            # Compute l_i = ||u_{i+1} - u_{i+2}|| for i=1,2,3
            l1 = np.linalg.norm(u2 - u3)
            l2 = np.linalg.norm(u3 - u1)
            l3 = np.linalg.norm(u1 - u2)

            theta1 = 2 * self.safe_arcsin(l1 / 2)
            theta2 = 2 * self.safe_arcsin(l2 / 2)
            theta3 = 2 * self.safe_arcsin(l3 / 2)

            theta = [theta1, theta2, theta3]
            h = sum(theta) / 2

            if (np.pi - h) < epsilon:
                w1 = np.sin(theta1) * l2 * l3
                w2 = np.sin(theta2) * l1 * l3
                w3 = np.sin(theta3) * l1 * l2
                w_total = w1 + w2 + w3
                f_x = (w1 * f1 + w2 * f2 + w3 * f3) / w_total
                return f_x

            # Compute c_i = (2 * sin(h) * sin(h - theta_i)) / (sin(theta_{i+1}) * sin(theta_{i+2})) ) - 1
            c = []
            for i in range(3):
                theta_i = theta[i]
                theta_ip1 = theta[(i + 1) % 3]
                theta_ip2 = theta[(i + 2) % 3]
                numerator = 2 * np.sin(h) * np.sin(h - theta_i)
                denominator = np.sin(theta_ip1) * np.sin(theta_ip2)
                if denominator == 0:
                    c_i = -1  # To ensure stability
                else:
                    c_i = (numerator / denominator) - 1
                c.append(c_i)

            c = np.array(c)
            s = np.sign(np.linalg.det(np.stack([u1, u2, u3]))) * np.sqrt(1-c**2)

            if np.any(abs(s) <= epsilon):
                continue

            w = []
            for i in range(3):
                d_i = distances[tri[i]]
                denominator = np.sin(theta[(i + 1) % 3]) * s[(i+2) % 3] * d_i
                numerator = theta[i] - c[(i+1)%3] * theta[(i+2)%3] - c[(i+2)%3] * theta[(i+1)%3]
                if denominator == 0:
                    w_j = 0
                else:
                    w_j = numerator / denominator
                w.append(w_j)

            # Accumulate totalF and totalW
            totalF += w[0] * f1 + w[1] * f2 + w[2] * f3
            totalW += w[0] + w[1] + w[2]

        f_x = totalF / totalW
        return f_x
    
    def compute_fxs(self, xs, epsilon=1e-6):
        """
        Optimized and partially vectorized version of compute_fxs.
        """

        projected_xs = self.projector.project_points_parallel(xs)

        # Compute distances between all xs and vertices
        # Shape: (n_points, n_vertices)
        distances = np.linalg.norm(
            self.vertices[np.newaxis, :, :] - projected_xs[:, np.newaxis, :], axis=2
        )

        # Find close vertices for all xs
        close_mask = distances < epsilon  # Shape: (n_points, n_vertices)
        close_indices = np.argmax(close_mask, axis=1)  # Closest vertex per point

        # Allocate result array
        results = np.zeros((xs.shape[0], self.f_values.shape[1]))

        # Handle points near vertices
        is_close = np.any(close_mask, axis=1)
        results[is_close] = self.f_values[close_indices[is_close]]
        if np.all(is_close):
            return results

        not_done = ~is_close
        u_vectors = np.zeros((xs.shape[0], self.vertices.shape[0], 3))
        u_vectors[not_done] = (self.vertices[np.newaxis, :, :] - projected_xs[not_done, np.newaxis, :]) / (distances[not_done, :, np.newaxis])

        totalF = np.zeros((xs.shape[0], self.f_values.shape[1]))
        totalW = np.zeros(xs.shape[0])

        for tri in self.triangles:

            idx1, idx2, idx3 = tri 
            f1, f2, f3 = self.f_values[idx1], self.f_values[idx2], self.f_values[idx3] 

            u1 = u_vectors[:, idx1]
            u2 = u_vectors[:, idx2]
            u3 = u_vectors[:, idx3]

            # Compute l_i for all points
            l1 = np.linalg.norm(u2 - u3, axis=1)
            l2 = np.linalg.norm(u3 - u1, axis=1)
            l3 = np.linalg.norm(u1 - u2, axis=1)

            theta1 = 2 * self.safe_arcsin(l1 / 2)
            theta2 = 2 * self.safe_arcsin(l2 / 2)
            theta3 = 2 * self.safe_arcsin(l3 / 2)
            theta = np.stack([theta1, theta2, theta3], axis=1)

            h = np.sum(theta, axis=1) / 2
            mask = (np.pi - h < epsilon) & not_done
            if np.any(mask):
                w1 = np.sin(theta1[mask]) * l2[mask] * l3[mask]
                w2 = np.sin(theta2[mask]) * l1[mask] * l3[mask]
                w3 = np.sin(theta3[mask]) * l1[mask] * l2[mask]
                w_total = w1 + w2 + w3
                results[mask] = (w1[:,np.newaxis] * f1[np.newaxis, :]+ 
                                 w2[:, np.newaxis] * f2[np.newaxis, :] + 
                                 w3[:, np.newaxis] * f3[np.newaxis, :])
                results[mask] /= w_total[:, np.newaxis]
                not_done ^= mask

            if not np.any(not_done):
                break

            c = np.zeros((xs.shape[0], 3))
            for i in range(3):
                theta_i = theta[:, i]
                theta_ip1 = theta[:, (i + 1) % 3]
                theta_ip2 = theta[:, (i + 2) % 3]
                numerator = 2 * np.sin(h) * np.sin(h - theta_i)
                denominator = np.sin(theta_ip1) * np.sin(theta_ip2)
                c[not_done, i] = numerator[not_done] / denominator[not_done] - 1

            det_values = np.linalg.det(np.stack([u1, u2, u3], axis=1))  # Shape: (n,)
            s = np.sign(det_values)[:, np.newaxis] * np.sqrt(1 - c**2)

            w = []
            for i in range(3):
                d_i = distances[:, tri[i]]
                denominator = np.sin(theta[:, (i + 1) % 3]) * s[:, (i+2) % 3] * d_i
                numerator = theta[:, i] - c[:, (i+1)%3] * theta[:, (i+2)%3] - c[:, (i+2)%3] * theta[:, (i+1)%3]
                w_i = np.zeros_like(numerator)
                w_i[not_done] = numerator[not_done] / denominator[not_done]
                w.append(w_i)

            totalF += np.outer(w[0], f1) + np.outer(w[1], f2) + np.outer(w[2], f3)    
            totalW += w[0] + w[1] + w[2]

        results[not_done] = totalF[not_done] / totalW[not_done, np.newaxis]
        return results
    
def get_mvc_of_pixels(image, pallete):
    N, M, _ = image.shape
    query = image.reshape((N*M, 3))
    mvc = MVC(pallete)
    coords = mvc.compute_fxs(query)
    return coords.reshape((N, M, pallete.shape[0]))

def construct_image_from_mvc(mvc_coords, pallete):
    return np.einsum('mnl,lp->mnp', mvc_coords, pallete)

if __name__ == "__main__":
    
    import time

    hull_vertices = np.array([
        [0.19761568, 0.2779199,  0.30872461],
        [0.35672523, 0.99113141, 0.50231554],
        [0.33906182, 0.57454616, 0.88112569],
        [0.89005058, 0.6739143,  0.36623488],
        [0.31112575, 0.6286831,  0.35056347],
        [0.82094264, 0.39706136, 0.48109977],
        [0.2894192,  0.00547371, 0.68187885],
        [0.77351632, 0.63527035, 0.14662607],
        [0.92219756, 0.4549181,  0.11490841],
        [0.03631562, 0.09841079, 0.63888051]
    ])
    mvc = MVC(hull_vertices)

    query_points = np.random.rand(1000000, 3)

    start_time2 = time.time()
    result2 = mvc.compute_fxs(query_points)
    end_time2 = time.time()

    print(end_time2 - start_time2)

