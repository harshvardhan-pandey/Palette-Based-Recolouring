import argparse
import numpy as np
from projection import Projector
from PIL import Image
import os
import time

class MVC:

    def __init__(self, palette_points):
        self.palette_points = palette_points.copy()
        self.f_values = np.eye(palette_points.shape[0], dtype=np.float64)
        self.projector = Projector(self.palette_points)
        self.triangles = self.projector.hull.simplices
        self.vertices = self.palette_points[self.projector.hull.vertices]

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
    
def load_from_file(filename: str):
    if not os.path.exists(filename):
        print("Loading palette failed")
        return False
    
    with open(filename, "r") as infile:
        lines = infile.readlines()
        
    lines = [line.split() for line in lines]
    points = np.array((0, 3), dtype=np.float32)
    points = np.empty((0, 3), dtype=np.float32)
    for l in lines:
        if l[0] == "v":
            points = np.vstack([points, np.array(l[1:]).astype(float)])
    return points

def show_palette(palette, opt, width = 100, height = 100):
    N, _ = palette.shape
    image = np.zeros((height, N*width, 3))
    for i in range(N):
        colour = palette[i]
        image[:, i*width:(i+1)*width] = colour
    
    palette_img = Image.fromarray((image * 255).astype(np.uint8), mode="RGB")
    palette_img.save(opt.out + "cur_palette.png")

    
def get_mvc_of_pixels(image, palette):
    N, M, _ = image.shape
    query = image.reshape((N*M, 3))
    mvc = MVC(palette)
    coords = mvc.compute_fxs(query)
    return coords.reshape((N, M, palette.shape[0]))

def construct_image_from_mvc(mvc_coords, palette):
    return np.einsum('mnl,lp->mnp', mvc_coords, palette)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="path to image")
    parser.add_argument("--pal", help="path to palette")
    parser.add_argument("--out", help="path to output folder")
    opt = parser.parse_args()
    
    img = np.array(Image.open(opt.img)) / 255
    palette = load_from_file(opt.pal)
    
    weights = get_mvc_of_pixels(img, palette)
    
    while True:
        show_palette(palette, opt)
        inp = input("Do you wish to change the palette? [y/n]")
        if inp == "n":
            break
        idx = int(input("Enter index of color to change (0-indexing) "))
        r = int(input("Enter red value for new color (0 - 255) ")) / 255
        g = int(input("Enter green value for new color (0 - 255) ")) / 255
        b = int(input("Enter blue value for new color (0 - 255) ")) / 255
        palette[idx] = np.array([r, g, b])
    
    new_img = construct_image_from_mvc(weights, palette)
    new_img = Image.fromarray((new_img * 255).astype(np.uint8), mode="RGB")
    new_img.save(opt.out + "new_albedo.png")