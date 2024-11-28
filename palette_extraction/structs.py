import numpy as np
import os


def normalize(v):
    return v / np.linalg.norm(v)


class Ray:
    def __init__(self, s, d) -> None:
        self.origin = s
        self.direction = normalize(d)

    def point_at_parameter(self, t):
        return self.origin + self.direction * t


class Triangle:
    def __init__(self, a, b, c) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.norm = normalize(np.cross(b - a, c - a))
        self.d = -np.dot(self.norm, a)

    def check_side(a, b, c, p):
        ab = b - a
        ac = c - a
        ap = p - a
        v1 = np.cross(ab, ac)
        v2 = np.cross(ab, ap)
        return np.dot(v1, v2) >= 0

    def in_tri(self, p):
        res = True
        sides = np.array([self.a, self.b, self.c])
        for _ in range(3):
            res = res and Triangle.check_side(sides[0], sides[1], sides[2], p)
            sides = np.roll(sides, 1)
        return res

    def intersection_parameter(self, ray: Ray):
        eps = 1e-8
        dot_product = np.dot(self.norm, ray.direction)
        if np.abs(dot_product) < eps:
            return 0
        t = -(self.d + np.dot(self.norm, ray.origin)) / (dot_product + eps)
        if t > eps:
            point = ray.point_at_parameter(t)
            inter = self.in_tri(point)
            if inter:
                return t
        return 0


class Mesh:
    def __init__(self) -> None:
        self.vertices = np.array((0, 3), dtype=np.float32)
        self.faces = np.array((0, 3), dtype=np.int32)

    def clear(self):
        self.vertices = np.empty((0, 3), dtype=np.float32)
        self.faces = np.empty((0, 3), dtype=np.int32)

    def vertex_num(self) -> int:
        return self.vertices.shape[0]

    def face_num(self) -> int:
        return self.faces.shape[0]
    
    def copy(self):
        new_mesh = Mesh()
        new_mesh.vertices = self.vertices.copy()
        new_mesh.faces = self.faces.copy()
        return new_mesh

    def load_from_file(self, filename: str) -> bool:
        self.clear()

        if not os.path.exists(filename):
            print("Loading failed")
            return False

        with open(filename, "r") as infile:
            lines = infile.readlines()
        lines = [line.split() for line in lines]
        for l in lines:
            if l[0] == "v":
                self.vertices = np.vstack(
                    [self.vertices, np.array(l[1:]).astype(float)]
                )
            elif l[0] == "f":
                self.faces = np.vstack([self.faces, np.array(l[1:]).astype(int)])
        return True

    def construct_triangles(self):
        triangles = []
        for i in range(self.face_num()):
            a = self.vertices[self.faces[i][0]]
            b = self.vertices[self.faces[i][1]]
            c = self.vertices[self.faces[i][2]]
            triangles.append(Triangle(a, b, c))
        return np.array(triangles)

    def save_to_file(self, filename: str) -> bool:
        try:
            f = open(filename, "w")
        except:
            print("Loading failed")
            return False
        finally:
            f.close()

        with open(filename, "w") as outfile:
            for v in self.vertices:
                outfile.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for f in self.faces:
                outfile.write(f"f {f[0]} {f[1]} {f[2]}\n")
        return True

    def is_inside(self, triangles, pt) -> bool:
        r = Ray(pt, np.array([1, 0, 1]))
        count = 0
        for i in range(len(triangles)):
            t = triangles[i]
            if t.intersection_parameter(r) > 0:
                count += 1
        return count % 2 == 1

    def is_inside_batch(self, triangles, points):
        ray_dir = np.array([1, 0, 1])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        count = np.zeros((len(points)))
        for i in range(len(triangles)):
            tri = triangles[i]

            eps = 1e-8
            dot_product = np.dot(tri.norm, ray_dir)
            if np.abs(dot_product) < eps:
                continue

            t = -(tri.d + np.dot(tri.norm, points.T)) / (dot_product + eps)
            new_points = points + t[:, np.newaxis] * ray_dir
            res = np.array([True for _ in range(len(new_points))])
            sides = np.array([tri.a, tri.b, tri.c])
            for _ in range(3):
                a = sides[0]
                b = sides[1]
                c = sides[2]
                ab = b - a
                ac = c - a
                ap = new_points - a
                v1 = np.cross(ab, ac)
                v2 = np.cross(ab, ap)
                res = np.logical_and(res, (np.dot(v2, v1) >= 0))

                sides = np.roll(sides, 1, axis=0)
            count += (np.logical_and(t > eps, np.logical_and(res, t > 0))).astype(int)
        return count % 2 == 1


class PointDist:
    def __init__(self, point, dist) -> None:
        self.point = np.array(point)
        self.dist = dist

    def __lt__(self, other):
        return self.dist < other.dist


class Data:
    def __init__(
        self,
        center_point,
        index: int,
        mesh: Mesh,
        points,
        lambda_fac: float,
        num_nearest_neighbours: int,
        center_point_option: int,
        unique: int,
    ) -> None:
        self.center_point = center_point
        self.index = index
        self.mesh = mesh
        self.points = points
        self.lambda_fac = lambda_fac
        self.num_nearest_neighbours = num_nearest_neighbours
        self.center_point_option = center_point_option
        self.unique = unique


class Result:
    def __init__(
        self, lambda_fac, reconstruct_error, represent_error, total_error
    ) -> None:
        self.lambda_fac = lambda_fac
        self.total_error = total_error
        self.reconstruct_error = reconstruct_error
        self.represent_error = represent_error


def closest_point_on_triangle(triangle, source_position):
    edge0 = triangle[1] - triangle[0]
    edge1 = triangle[2] - triangle[0]
    v0 = triangle[0] - source_position

    a = np.dot(edge0, edge0)
    b = np.dot(edge0, edge1)
    c = np.dot(edge1, edge1)
    d = np.dot(edge0, v0)
    e = np.dot(edge1, v0)

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    if s + t < det:
        if s < 0.0:
            if t < 0.0:
                if d < 0.0:
                    s = np.clip(-d / a, 0.0, 1.0)
                    t = 0.0
                else:
                    s = 0.0
                    t = np.clip(-e / c, 0.0, 1.0)
            else:
                s = 0.0
                t = np.clip(-e / c, 0.0, 1.0)
        elif t < 0.0:
            s = np.clip(-d / a, 0.0, 1.0)
            t = 0.0
        else:
            invDet = 1.0 / det
            s *= invDet
            t *= invDet
    else:
        if s < 0.0:
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2 * b + c
                s = np.clip(numer / denom, 0.0, 1.0)
                t = 1.0 - s
            else:
                t = np.clip(-e / c, 0.0, 1.0)
                s = 0.0
        elif t < 0.0:
            if a + d > b + e:
                numer = c + e - b - d
                denom = a - 2 * b + c
                s = np.clip(numer / denom, 0.0, 1.0)
                t = 1.0 - s
            else:
                s = np.clip(-e / c, 0.0, 1.0)
                t = 0.0
        else:
            numer = c + e - b - d
            denom = a - 2 * b + c
            s = np.clip(numer / denom, 0.0, 1.0)
            t = 1.0 - s

    return triangle[0] + s * edge0 + t * edge1


def nearest_point(x, triangles, vertices):
    dis = []
    close_points = []
    for tri in triangles:
        tri_vertices = [vertices[tri[i]] for i in range(3)]
        close_point = closest_point_on_triangle(tri_vertices, x)
        dis.append(np.square(np.linalg.norm(x - close_point)))
        close_points.append(close_point)
    idx = np.argmin(dis)
    return close_points[idx]
