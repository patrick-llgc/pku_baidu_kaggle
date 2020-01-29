import json
import numpy as np
import open3d as o3d

from utils import get_avg_size
from render_and_compare import load_model_files

TriangleMesh = o3d.geometry.TriangleMesh
Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
draw_geometries = o3d.visualization.draw_geometries

basedir = '/Users/pliu/github/pku_baidu_kaggle/data//'

avg_size_dict = get_avg_size()
vertices, triangles = load_model_files(avg_size_dict['3x']['model'], basedir=basedir)
vertices[:, 1] = -vertices[:, 1] ## y is pointing downward

# vis
mesh = TriangleMesh()
mesh.vertices = Vector3dVector(vertices)
mesh.triangles = Vector3iVector(triangles)
mesh.paint_uniform_color([0, 1, 0])

# car_v2 = np.copy(vertices)
# car_v2[:, 2] += car_v2[:, 2].max()*3
# mesh2 = TriangleMesh()
# mesh2.vertices = Vector3dVector(car_v2)
# mesh2.triangles = Vector3iVector(triangles)
# mesh2.paint_uniform_color([0, 0, 1])
# draw_geometries([mesh, mesh2])

print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
# print(np.asarray(mesh.triangle_normals))
draw_geometries([mesh])