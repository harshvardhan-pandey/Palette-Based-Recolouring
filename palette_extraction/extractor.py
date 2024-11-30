import argparse
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import *
from math import *
import cvxopt   
import PIL.Image as Image  
import sys    

######***********************************************************************************************

#### 3D case: use method in paper: "Progressive Hulls for Intersection Applications"
#### also using trimesh.py interface from yotam gingold    
    
from trimesh import TriMesh

def get_hull_vertices_faces(hull, normalized = False):

    hvertices = hull.points[hull.vertices]
    points_index = -1 * np.ones(hull.points.shape[0], dtype=np.int32)  # Using int32 instead of default int
    points_index[hull.vertices] = np.arange(len(hull.vertices))
    
    hfaces = points_index[hull.simplices]
    vertices = hvertices[hfaces]
    edges1 = vertices[:, 1] - vertices[:, 0]
    edges2 = vertices[:, 2] - vertices[:, 0]
    face_normals = np.cross(edges1, edges2)
    
    dots = np.sum(hull.equations[:, :3] * face_normals, axis=1)
    flip_mask = dots < 0
    hfaces[flip_mask, :2] = hfaces[flip_mask, :2][:, ::-1]

    if normalized:
        hvertices /= 255

    return hvertices, hfaces

def convert_convexhull_to_trimesh(hull):
    result = TriMesh()
    hvertices, hfaces = get_hull_vertices_faces(hull)
    result.vs = hvertices
    result.faces = hfaces    
    return result

def write_convexhull_into_obj_file(hull, output_rawhull_obj_file):

    hvertices, hfaces = get_hull_vertices_faces(hull, normalized=True)
            
    myfile=open(output_rawhull_obj_file,'w')
    for index in range(hvertices.shape[0]):
        myfile.write('v '+str(hvertices[index][0])+' '+str(hvertices[index][1])+' '+str(hvertices[index][2])+'\n')
    for index in range(hfaces.shape[0]):
        myfile.write('f '+str(hfaces[index][0])+' '+str(hfaces[index][1])+' '+str(hfaces[index][2])+'\n')
    myfile.close()


def edge_normal_test(vertices, faces, old_face_index_list, v0_ind, v1_ind):
    selected_old_face_list=[]
    central_two_face_list=[]
    
    for index in old_face_index_list:
        face=faces[index]
        face_temp=np.array(face).copy()
        face_temp=list(face_temp)
        
        if v0_ind in face_temp:
            face_temp.remove(v0_ind)
        if v1_ind in face_temp:
            face_temp.remove(v1_ind)
        if len(face_temp)==2:  ### if left 2 points, then this face is what we need.
            selected_old_face=[np.asarray(vertices[face[i]]) for i in range(len(face))]
            selected_old_face_list.append(np.asarray(selected_old_face))
        if len(face_temp)==1: ##### if left 1 points, then this face is central face.
            central_two_face=[np.asarray(vertices[face[i]]) for i in range(len(face))]
            central_two_face_list.append(np.asarray(central_two_face))
            
    assert( len(central_two_face_list)==2 )
    if len(central_two_face_list)+len(selected_old_face_list)!=len(old_face_index_list):
        print( 'error!!!!!!' )
    
    central_two_face_normal_list=[]
    neighbor_face_dot_normal_list=[]
    
    for face in central_two_face_list:
        n=np.cross(face[1]-face[0], face[2]-face[0])
        n=n/np.sqrt(np.dot(n,n))
        central_two_face_normal_list.append(n)
        
    avg_edge_normal=np.average(np.array(central_two_face_normal_list),axis=0)
    
    for face in selected_old_face_list:
        n=np.cross(face[1]-face[0], face[2]-face[0])
        neighbor_face_dot_normal_list.append(np.dot(avg_edge_normal,n))
    
    if (np.array(neighbor_face_dot_normal_list)>=0.0-1e-5).all():
        return 1
    else:
        return 0


        
def compute_tetrahedron_volume(face, point):
    n=np.cross(face[1]-face[0], face[2]-face[0])
    return abs(np.dot(n, point-face[0]))/6.0

#### this is different from function: remove_one_edge_by_finding_smallest_adding_volume(mesh)
#### return a new mesh (using trimesh.py)

from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class EdgeCollapseResult:
    index: int
    volume: float
    point: np.ndarray
    v1_ind: int
    v2_ind: int

def mesh_edge_removal(mesh):

    edges = np.asarray(mesh.get_edges())
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vs)
    
    # Pre-configure solver options
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    
    best_result = find_best_edge_collapse(mesh, edges, faces, vertices)
    
    if best_result is None:
        print('all fails')
        return mesh
        
    # Apply the best edge collapse
    apply_edge_collapse(mesh, best_result)
    return mesh

def find_best_edge_collapse(mesh, edges: np.ndarray, faces: np.ndarray, 
                          vertices: np.ndarray) -> Optional[EdgeCollapseResult]:
    """Find the best edge to collapse based on volume criteria."""
    results = []
    
    for edge_idx, (v1_ind, v2_ind) in enumerate(edges):
        # Get related faces efficiently using set operations
        face_indices = set(mesh.vertex_face_neighbors(v1_ind)) | set(mesh.vertex_face_neighbors(v2_ind))
        related_faces = faces[list(face_indices)]
        
        # Setup optimization problem
        A, b, c = setup_optimization_problem(vertices, related_faces)
        
        # Solve optimization problem
        result = solve_edge_collapse(A, b, c)
        if result is None:
            continue
            
        # Calculate volume
        new_point = result
        volumes = calculate_face_volumes(vertices[related_faces], new_point)
        total_volume = np.sum(volumes)
        
        results.append(EdgeCollapseResult(
            index=edge_idx,
            volume=total_volume,
            point=new_point,
            v1_ind=v1_ind,
            v2_ind=v2_ind
        ))
    
    if not results:
        return None
        
    return min(results, key=lambda x: x.volume)

def setup_optimization_problem(vertices: np.ndarray, related_faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Setup the optimization problem matrices."""
    face_points = vertices[related_faces]
    
    # Vectorized normal calculation
    v0 = face_points[:, 0]
    v1 = face_points[:, 1]
    v2 = face_points[:, 2]
    
    # Calculate normals in a vectorized way
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms  # Normalize normals
    
    A = -normals
    b = -np.sum(normals * v0, axis=1)
    c = np.sum(normals, axis=0)
    
    return A, b, c

def solve_edge_collapse(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[np.ndarray]:
    """Solve the optimization problem for edge collapse."""
    try:
        res = cvxopt.solvers.lp(
            cvxopt.matrix(c), 
            cvxopt.matrix(A), 
            cvxopt.matrix(b),
            solver='glpk'
        )
        
        if res['status'] == 'optimal':
            return np.asarray(res['x'], dtype=np.float64).squeeze()
        return None
    except:
        return None

def calculate_face_volumes(faces: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Vectorized calculation of tetrahedron volumes."""
    v0 = faces[:, 0]
    v1 = faces[:, 1]
    v2 = faces[:, 2]
    
    # Calculate volume using triple product
    volumes = np.abs(np.sum(
        np.cross(v1 - v0, v2 - v0) * (point - v0),
        axis=1
    )) / 6
    return volumes

def apply_edge_collapse(mesh, result: EdgeCollapseResult) -> None:
    """Apply the edge collapse operation to the mesh."""
    v1_ind, v2_ind = result.v1_ind, result.v2_ind
    
    # Get affected faces
    face_indices = set(mesh.vertex_face_neighbors(v1_ind)) | set(mesh.vertex_face_neighbors(v2_ind))
    related_faces = [mesh.faces[idx] for idx in face_indices]
    
    # Remove vertices and update indices
    old2new = mesh.remove_vertex_indices([v1_ind, v2_ind])
    new_vertex_index = len(old2new[old2new != -1])
    
    # Process faces efficiently using numpy operations
    new_faces = process_faces(related_faces, v1_ind, v2_ind, new_vertex_index, old2new)
    
    # Update mesh
    mesh.vs = list(mesh.vs)
    mesh.vs.append(result.point)
    if mesh.faces is not None:
        mesh.faces = np.array(mesh.faces.tolist().extend(new_faces))
    mesh.topology_changed()

def process_faces(faces, v1_ind, v2_ind, new_vertex_index, old2new):
    """Process faces efficiently using numpy operations."""
    faces_array = np.array(faces)
    mask = (faces_array == v1_ind) | (faces_array == v2_ind)
    
    new_faces = []
    for face in faces_array:
        new_face = np.where(mask[len(new_faces)], new_vertex_index, 
                          [old2new[x] for x in face])
        if len(np.unique(new_face)) == len(new_face):
            new_faces.append(new_face.tolist())
    
    return new_faces

def get_simplified_hull(image, E_vertice_num = 4, N = 500):

    image = image.reshape((-1,3))
    hull = ConvexHull(image)
    origin_hull=hull

    mesh=convert_convexhull_to_trimesh(origin_hull)

    for _ in range(N):
        
        old_num=len(mesh.vs)
        mesh=mesh_edge_removal(mesh)
        newhull=ConvexHull(mesh.vs)
        mesh = convert_convexhull_to_trimesh(newhull)

        if len(mesh.vs)==old_num or len(mesh.vs)<=E_vertice_num:
            break

    newhull=ConvexHull(mesh.vs)

    return newhull

def get_simplified_hull_vertices_faces(image, E_vertice_num = 4, N = 500, normalized = True):
    return get_hull_vertices_faces(get_simplified_hull(image, E_vertice_num, N), normalized)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="path to input image")
    parser.add_argument("--out", help="path to output folder")
    parser.add_argument("--n_col", help="number of colours in palette", default=4)
    opt = parser.parse_args()

    input_image_path = opt.img
    output_folder = opt.out
    E_vertice_num = int(opt.n_col)
    images = np.asarray(Image.open(input_image_path).convert('RGB'), dtype=np.float64)

    from time import process_time as clock
    
    start_time = clock()
    hull = get_simplified_hull(images, E_vertice_num=E_vertice_num)
    end_time = clock()

    write_convexhull_into_obj_file(hull, output_folder + "original_palette.obj")
    print( 'time: ', end_time-start_time )