import argparse
import cv2
import sys
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from structs import *


def compute_specific_center_point(
    mesh: Mesh, inside_points, num_nearest_neighbour: int, vertex_index: int
):
    # print("len of inside ", len(inside_points))
    vertex = mesh.vertices[vertex_index]
    dist = np.linalg.norm(inside_points - vertex, axis=1)
    idx = np.argsort(dist)[:num_nearest_neighbour]
    center = np.mean(inside_points[idx], axis=0)
    return center


def compute_center_points(mesh: Mesh, inside_points, num_nearest_neighbour: int):
    center_points = []
    for l in range(mesh.vertex_num()):
        center_points.append(
            compute_specific_center_point(mesh, inside_points, num_nearest_neighbour, l)
        )
    return np.array(center_points)


def find_nearest(mesh: Mesh, point):
    dist = np.linalg.norm(mesh.vertices - point, axis=1)
    parent = np.argmin(dist)
    distance = dist[parent]
    return (parent, distance)


def compute_center_points_unique(
    mesh: Mesh, inside_points, num_nearest_neighbours: int
):
    center_points = []
    included_points = [[] for _ in range(mesh.vertex_num())]

    for pt in inside_points:
        res = find_nearest(mesh, pt)
        included_points[res[0]].append(PointDist(pt, res[1]))

    for i in range(mesh.vertex_num()):
        sorted_inc_points = np.sort(included_points[i])[:num_nearest_neighbours]
        if len(sorted_inc_points) > 0:
            points = np.array([pt.point for pt in sorted_inc_points])
            center = np.sum(points, axis=0)
            center = center / float(num_nearest_neighbours)
            center_points.append(center)
        else:
            center_points.append(np.array([0.0, 0.0, 0.0]))
    return np.array(center_points)


def compute_outside_points(mesh: Mesh, points):
    triangles = mesh.construct_triangles()
    # is_inside = mesh.is_inside_batch(triangles, points)
    delaunay_tri = Delaunay(mesh.vertices)
    is_inside = delaunay_tri.find_simplex(points) >= 0
    inside_points = np.array([points[i] for i in range(len(points)) if is_inside[i]])
    outside_points = np.array(
        [points[i] for i in range(len(points)) if not is_inside[i]]
    )
    return inside_points, outside_points


def outside_points_distance(mesh: Mesh, outside_points) -> float:
    nearest_points = np.array(
        [
            nearest_point(outside_points[i], mesh.faces, mesh.vertices)
            for i in range(len(outside_points))
        ]
    )
    distances = np.linalg.norm(nearest_points - outside_points, axis=1)
    return np.sum(distances)


def compute_loss_function_base(
    mesh: Mesh, outside_points, num_total_points, center_nn_points, lambda_fac
):
    if len(outside_points) == 0:
        reconstruct_error = 0.0
    else:
        reconstruct_error = (
            outside_points_distance(mesh, outside_points) / num_total_points
        )
    represent_error = np.sum(np.linalg.norm(mesh.vertices - center_nn_points, axis=1))
    represent_error /= mesh.vertex_num()
    total_error = reconstruct_error * lambda_fac + represent_error
    return Result(lambda_fac, reconstruct_error, represent_error, total_error)


def compute_loss_function(
    mesh: Mesh,
    points,
    lambda_fac: float,
    num_nearest_neighbours: int,
    option: int,
    unique: int,
):
    inside_points, outside_points = compute_outside_points(mesh, points)

    if unique == 1:
        if option == 1:
            n = min(num_nearest_neighbours, len(inside_points))
            center_points = compute_center_points_unique(mesh, inside_points, n)
        else:
            center_points = compute_center_points_unique(
                mesh, points, num_nearest_neighbours
            )
    else:
        if option == 1:
            n = min(num_nearest_neighbours, len(inside_points))
            center_points = compute_center_points(mesh, inside_points, n)
        else:
            center_points = compute_center_points(mesh, points, num_nearest_neighbours)

    return compute_loss_function_base(
        mesh, outside_points, len(points), center_points, lambda_fac
    )


def cost_function(x, data: Data) -> float:
    # x - current k to be considered
    # data -
    # list of v_c (center_point)
    # index of v to be changes (index)
    # mesh
    # points
    # lambda
    # M used to calculate v_c (num_nearest_neighbours)
    # center_point_option - 1 (use only inside points) or 0 (use all points)
    # unique - 1 (make set of neighbours disjoint) or 0 (not)

    # calculate new point with current k
    target_dist = x[0]
    target_point = (1 - target_dist) * data.center_point[
        data.index
    ] + target_dist * data.mesh.vertices[data.index]

    # print("k: ", target_dist)
    # print("ori c pt: ", data.center_point[data.index])
    # print("ori m pt: ", data.mesh.vertices[data.index])
    # print("k pt: ", target_point)

    # adjust the mesh with new point
    new_mesh = data.mesh.copy()
    new_mesh.vertices[data.index] = target_point

    # calculate the center points wrt new mesh
    inside_points, outside_points = compute_outside_points(new_mesh, data.points)
    num_nearest_neighbours = data.num_nearest_neighbours
    center_points = data.center_point.copy()

    if data.unique == 1:
        if data.center_point_option == 1:
            n = min(num_nearest_neighbours, len(inside_points))
            center_points[data.index] = compute_center_points_unique(
                new_mesh, inside_points, n
            )[data.index]
        else:
            center_points[data.index] = compute_center_points_unique(
                new_mesh, data.points, num_nearest_neighbours
            )[data.index]
    else:
        if data.center_point_option == 1:
            n = min(num_nearest_neighbours, len(inside_points))
            center_points[data.index] = compute_specific_center_point(
                new_mesh, inside_points, n, data.index
            )
        else:
            center_points[data.index] = compute_specific_center_point(
                new_mesh, data.points, num_nearest_neighbours, data.index
            )

    # compute loss based on new mesh
    result = compute_loss_function_base(
        new_mesh,
        outside_points,
        len(data.points),
        center_points,
        data.lambda_fac,
    )

    return result.total_error


if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="original picture")
    parser.add_argument("--obj", help=".obj file for initial convex hull")
    parser.add_argument("--out", help="path to output folder")
    parser.add_argument("--sample", help="number of samples", default=200)
    parser.add_argument("--option", help="center point selection method", default=1)
    parser.add_argument(
        "--ratio", help="ratio between neighbour point and random points", default=0.001
    )
    parser.add_argument("--lambda_f", help="lambda factor in cost function", default=20)
    parser.add_argument("--iter", help="number of iterations", default=2)
    parser.add_argument("--unique", help="unique parent of point", default=0)
    opt = parser.parse_args()

    num_sample = int(opt.sample)
    center_point_option = int(opt.option)
    neighbour_point_ratio = float(opt.ratio)
    lambda_fac = float(opt.lambda_f)
    iterations = int(opt.iter)
    unique = int(opt.unique)

    print(f"image: {opt.img}")
    print(f"original palette: {opt.obj}")

    # initial palette - the original convex hull
    mesh = Mesh()
    if not mesh.load_from_file(opt.obj):
        print(f"Load from {opt.obj} failed\n")
        sys.exit(0)

    # image
    img = cv2.imread(opt.img)
    if len(img) == 0 or len(img[0]) == 0:
        print(f"Load from {opt.img} failed\n")
        sys.exit(0)

    points = []
    seed = int(time.time())
    np.random.seed(seed)
    dis = lambda: np.random.uniform(0.0, 0.99)

    # number of samples to use, not all pixels
    sample_width = min(num_sample, img.shape[1])
    sample_height = min(num_sample, img.shape[0])
    ratio_width = img.shape[1] / sample_width
    ratio_height = img.shape[0] / sample_height

    # sampling the points
    for i in range(sample_height):
        for j in range(sample_width):
            x = min(int(np.floor((i + dis()) * ratio_height)), img.shape[0] - 1)
            y = min(int(np.floor((j + dis()) * ratio_width)), img.shape[1] - 1)
            pt = img[x, y]
            pt = pt[::-1]
            points.append(pt / 255)
    points = np.array(points)

    # use the neighbour_point_ratio to decide M - number of nearest neighbours to use
    # to calculate v_c - center of neighbouring points of point v
    num_nearest_neighbours = int(
        max(neighbour_point_ratio * sample_width * sample_height, 2)
    )
    refined_palette = mesh

    t_start = time.time()
    for z in range(iterations):
        for i in range(mesh.vertex_num()):
            # print(z, i)
            # calculate v_c for each v
            if unique == 1:
                center_points = compute_center_points_unique(
                    refined_palette, points, num_nearest_neighbours
                )
            else:
                center_points = compute_center_points(
                    refined_palette, points, num_nearest_neighbours
                )

            data = Data(
                center_points,
                i,
                refined_palette,
                points,
                lambda_fac,
                num_nearest_neighbours,
                center_point_option,
                unique,
            )

            ks = np.arange(0.0, 1.0, 0.1)
            cost = np.array([cost_function([k], data) for k in ks])
            mincost_idx = np.argmin(cost)
            minval = cost[mincost_idx]
            initial_guess = ks[mincost_idx]

            # print(ks)
            # print(cost)

            x0 = np.array([initial_guess])
            bounds = [(0, 1)]

            # print("optimize start")
            res = minimize(
                cost_function,
                x0,
                args=(data,),
                method="COBYLA",
                bounds=bounds,
                tol=1e-4,
            )
            refined_palette.vertices[data.index] = (1 - res.x) * data.center_point[
                data.index
            ] + res.x * data.mesh.vertices[data.index]

    t_end = time.time()
    print("time taken: ", t_end - t_start)

    refined_palette.save_to_file(opt.out + "refined_palette.obj")

    final_res = compute_loss_function(
        refined_palette,
        points,
        lambda_fac,
        num_nearest_neighbours,
        center_point_option,
        unique,
    )
    print("total loss: ", final_res.total_error)
    print("lambda: ", final_res.lambda_fac)
    print("reconstruct loss: ", final_res.reconstruct_error)
    print("representative error/sparse loss: ", final_res.represent_error)
