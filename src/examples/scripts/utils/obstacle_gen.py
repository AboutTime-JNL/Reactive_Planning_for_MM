import os

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import copy

coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)

source = (
    "/app/src/examples/assets/pcd"
)
source_obj = (
    "/app/src/examples/assets/obj"
)


# local minima
def get_shelf2_desk_shape():
    bookshelf_1 = [
        {"scale": [1.0, 0.60, 0.02], "base": [-0.5, 2.0, 0.5]},
        {"scale": [1.0, 0.60, 0.02], "base": [-0.5, 2.0, 1.0]},
        {"scale": [1.0, 0.60, 0.02], "base": [-0.5, 2.0, 1.5]},
        {"scale": [0.02, 0.60, 1.50], "base": [-0.52, 2.0, 0]},
        {"scale": [0.02, 0.60, 1.50], "base": [0.5, 2.0, 0]},
    ]

    bookshelf_2 = [
        {"scale": [1.0, 0.60, 0.02], "base": [-0.5, -2.6, 0.5]},
        {"scale": [1.0, 0.60, 0.02], "base": [-0.5, -2.6, 1.0]},
        {"scale": [1.0, 0.60, 0.02], "base": [-0.5, -2.6, 1.5]},
        {"scale": [0.02, 0.60, 1.50], "base": [-0.52, -2.6, 0]},
        {"scale": [0.02, 0.60, 1.50], "base": [0.5, -2.6, 0]},
    ]

    desk = [
        {"scale": [0.2, 0.2, 0.7], "base": [-1.5, -0.5, 0.0]},
        {"scale": [0.2, 0.2, 0.7], "base": [1.3, -0.5, 0.0]},
        {"scale": [0.2, 0.2, 0.7], "base": [-1.5, 0.3, 0.0]},
        {"scale": [0.2, 0.2, 0.7], "base": [1.3, 0.3, 0.0]},
        {"scale": [3.0, 1.0, 0.2], "base": [-1.5, -0.5, 0.7]},
    ]

    merged_mesh = o3d.geometry.TriangleMesh()
    trans = np.eye(4)

    for params in bookshelf_1 + bookshelf_2 + desk:
        scale = np.asarray(params["scale"])
        trans[:3, 3] = params["base"]
        box = o3d.geometry.TriangleMesh.create_box(
            width=scale[0], height=scale[1], depth=scale[2]
        )
        box.transform(trans)
        merged_mesh += box

    o3d.io.write_triangle_mesh(os.path.join(source_obj, "shelf2_desk.obj"), merged_mesh)
    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])

    num_points = 50000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "shelf2_desk.pcd"), point_cloud)


def get_wall_s_shape():
    thickness = 0.2
    length = 6.0
    width = 3.3
    height = 3.0

    bound = [
        {"scale": [length, thickness, height], "base": [-length / 2, width / 2, 0]},
        {"scale": [length, thickness, height], "base": [-length / 2, -width / 2 - thickness, 0], },
        {"scale": [thickness, width, height], "base": [length / 2, -width / 2, 0]},
        {"scale": [thickness, width, height], "base": [-length / 2 - thickness, -width / 2, 0], },
    ]

    wall_1 = [
        {"scale": [thickness, 1.9, height], "base": [1.0, -width / 2, 0.0]},
    ]

    wall_2 = [
        {"scale": [thickness, 1.9, height], "base": [-1.0, width / 2 - 2.1, 0.0]},
    ]

    merged_mesh = o3d.geometry.TriangleMesh()
    trans = np.eye(4)
    rot = Rotation.from_euler("xyz", [0, 0, -np.pi / 2]).as_matrix()

    for params in bound + wall_1 + wall_2:
        scale = np.asarray(params["scale"])
        trans[:3, 3] = params["base"]
        box = o3d.geometry.TriangleMesh.create_box(
            width=scale[0], height=scale[1], depth=scale[2]
        )
        box.transform(trans)
        box.rotate(rot, center=(0, 0, 0))
        merged_mesh += box

    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])
    num_points = 50000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "wall_s.pcd"), point_cloud)


def get_flower_shape():
    height = 2.0
    radius = 2.5
    thickness = 0.2

    center_cylinder = [
        {"scale": [0.8, height, 30], "base": [0, 0, height / 2]},
    ]

    other_cylinder = [
        {"scale": [0.4, height, 30], "base": [0, radius, height / 2]},
    ]

    wall = [
        {"scale": [thickness, radius, height], "base": [-thickness / 2, 0, 0.0]},
    ]

    length = 20.0
    width = 4.0

    bound = [
        {"scale": [length, thickness, 0.2], "base": [-length / 2, width / 2, 0]},
    ]

    merged_mesh = o3d.geometry.TriangleMesh()
    trans = np.eye(4)

    for params in center_cylinder:
        scale = np.asarray(params["scale"])
        trans[:3, 3] = params["base"]
        box = o3d.geometry.TriangleMesh.create_cylinder(
            radius=scale[0], height=scale[1]
        )
        box.transform(trans)
        merged_mesh += box

    for i in range(6):
        rot = Rotation.from_euler("xyz", [0, 0, i * np.pi / 3 + np.pi / 2]).as_matrix()
        for params in other_cylinder:
            scale = np.asarray(params["scale"])
            trans[:3, 3] = params["base"]
            box = o3d.geometry.TriangleMesh.create_cylinder(
                radius=scale[0], height=scale[1]
            )
            box.transform(trans)
            box.rotate(rot, center=(0, 0, 0))
            merged_mesh += box

        for params in wall:
            scale = np.asarray(params["scale"])
            trans[:3, 3] = params["base"]
            box = o3d.geometry.TriangleMesh.create_box(
                width=scale[0], height=scale[1], depth=scale[2]
            )
            box.transform(trans)
            box.rotate(rot, center=(0, 0, 0))
            merged_mesh += box

    rot = Rotation.from_euler("xyz", [0, 0, -np.pi / 2]).as_matrix()
    for params in bound:
        scale = np.asarray(params["scale"])
        trans[:3, 3] = params["base"]
        box = o3d.geometry.TriangleMesh.create_box(
            width=scale[0], height=scale[1], depth=scale[2]
        )
        box.transform(trans)
        box.rotate(rot, center=(0, 0, 0))
        merged_mesh += box

    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])
    num_points = 50000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "flower.pcd"), point_cloud)


# dynamic obstacle
def get_tie_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "tie.obj"), enable_post_processing=True
    )

    mesh.scale(0.001, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    center = mesh.get_center()
    mesh.translate((-center[0], -center[1], 0))

    merged_mesh = copy.deepcopy(mesh)

    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])
    num_points = 500000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "tie.pcd"), point_cloud)


def get_corridor_shape():
    length = 10
    width = 0.2
    height = 0.2
    # 间距
    distance = 2.8

    bound = [
        {"scale": [length, width, height], "base": [-length / 2, distance / 2, 0]},
        {"scale": [length, width, height], "base": [-length / 2, -distance / 2 - width, 0], },
    ]

    merged_mesh = o3d.geometry.TriangleMesh()
    trans = np.eye(4)
    rot = Rotation.from_euler("xyz", [0, 0, -np.pi / 2]).as_matrix()

    for params in bound:
        scale = np.asarray(params["scale"])
        trans[:3, 3] = params["base"]
        box = o3d.geometry.TriangleMesh.create_box(
            width=scale[0], height=scale[1], depth=scale[2]
        )
        box.transform(trans)
        box.rotate(rot, center=(0, 0, 0))
        merged_mesh += box

    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])
    num_points = 50000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "corridor.pcd"), point_cloud)

# multiple scene


def get_studio_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "studio.obj"), enable_post_processing=True
    )

    mesh.scale(0.03, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi))
    mesh_anti = copy.deepcopy(mesh).rotate(R, center=(0, 0, 0))
    mesh_anti = mesh_anti.translate((0.05, 0, 0))
    mesh_single = mesh + mesh_anti

    mesh_row = copy.deepcopy(mesh_single)
    for i in range(1):
        mesh_row += mesh_single.translate((0, 4, 0))

    merged_mesh = copy.deepcopy(mesh_row)
    for i in range(2):
        merged_mesh += mesh_row.translate((5, 0, 0))

    mesh_shelf = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "Shelves.obj"), enable_post_processing=True
    )
    mesh_shelf.scale(0.01, center=(0, 0, 0))
    R = mesh_shelf.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))
    mesh_shelf.rotate(R, center=(0, 0, 0))
    merged_mesh += mesh_shelf.translate((1, -2, 0))
    merged_mesh += mesh_shelf.translate((7, 0, 0))

    center = merged_mesh.get_center()
    merged_mesh.translate((-center[0], -center[1], 0))

    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])
    num_points = 100000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "studio.pcd"), point_cloud)


def get_warehouse_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "warehouse.obj"), enable_post_processing=True
    )

    mesh.scale(10, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate((0, 0, -0.4))

    merged_mesh = copy.deepcopy(mesh)

    center = merged_mesh.get_center()
    merged_mesh.translate((-center[0], -center[1], 0))

    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])
    num_points = 1000000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    points = np.asarray(point_cloud.points)
    mask = abs(points[:, 0]) < 4.7
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    points = np.asarray(point_cloud.points)
    mask = abs(points[:, 1]) < 11
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    points = np.asarray(point_cloud.points)
    mask = points[:, 2] > 0
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    points = np.asarray(point_cloud.points)
    mask = points[:, 2] < 2.0
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 0] > 0
    mask_2 = points[:, 0] < -0.3
    mask_3 = points[:, 1] < 1
    mask_4 = points[:, 1] > 4
    mask = mask_1 | mask_2 | mask_3 | mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "warehouse.pcd"), point_cloud)


def get_forest_shape():
    merged_mesh = o3d.geometry.TriangleMesh()
    trans = np.eye(4)

    for i in range(70):
        scale = np.random.rand(3)
        scale[0] = scale[0] * 0.5 + 0.25
        scale[1] = scale[1] * 0.5 + 0.25
        scale[2] = 2.0
        trans[:3, 3] = np.random.rand(3) * 14
        rot = np.random.rand(3) * np.pi
        rot[0] = 0
        rot[1] = 0
        trans[:3, :3] = Rotation.from_euler("xyz", rot).as_matrix()
        trans[2, 3] = 0
        box = o3d.geometry.TriangleMesh.create_box(
            width=scale[0], height=scale[1], depth=scale[2]
        )
        box.transform(trans)
        merged_mesh += box

    merged_mesh.translate((-7, -7, 0))

    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])
    num_points = 50000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 0] < 4.5
    mask_2 = points[:, 0] > 5.5
    mask_3 = points[:, 1] < 4.5
    mask_4 = points[:, 1] > 5.5
    mask = mask_1 | mask_2 | mask_3 | mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 0] < -5.5
    mask_2 = points[:, 0] > -4.5
    mask_3 = points[:, 1] < -5.5
    mask_4 = points[:, 1] > -4.5
    mask = mask_1 | mask_2 | mask_3 | mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "forest.pcd"), point_cloud)


def get_block_shape():
    merged_mesh = o3d.geometry.TriangleMesh()
    trans = np.eye(4)

    for i in range(90):
        scale = np.random.rand(3)
        scale[0] = scale[0] * 0.5 + 0.25
        scale[1] = scale[1] * 0.5 + 0.25
        scale[2] = 2.0
        trans[:3, 3] = np.random.rand(3) * 14
        trans[2, 3] = 0
        box = o3d.geometry.TriangleMesh.create_box(
            width=scale[0], height=scale[1], depth=scale[2]
        )
        box.transform(trans)
        merged_mesh += box

    merged_mesh.translate((-7, -7, 0))

    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])
    num_points = 50000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 0] < 4.5
    mask_2 = points[:, 0] > 5.5
    mask_3 = points[:, 1] < 4.5
    mask_4 = points[:, 1] > 5.5
    mask = mask_1 | mask_2 | mask_3 | mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 0] < -5.5
    mask_2 = points[:, 0] > -4.5
    mask_3 = points[:, 1] < -5.5
    mask_4 = points[:, 1] > -4.5
    mask = mask_1 | mask_2 | mask_3 | mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "block.pcd"), point_cloud)


def get_trees_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "trees.obj"), enable_post_processing=True
    )

    mesh.scale(0.6, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))

    merged_mesh = copy.deepcopy(mesh)

    center = merged_mesh.get_center()
    merged_mesh.translate((-center[0], -center[1], 0))

    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])
    num_points = 500000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "trees.pcd"), point_cloud)


def get_storage_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "storage.obj"), enable_post_processing=True
    )

    mesh.scale(0.001, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))

    merged_mesh = copy.deepcopy(mesh)
    center = merged_mesh.get_center()
    merged_mesh.translate((-center[0], -center[1], -0.2))

    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])
    num_points = 500000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 0] < -0.5
    mask_2 = points[:, 0] > 0.5
    mask_3 = points[:, 1] < 2.5
    mask_4 = points[:, 1] > 3.1
    mask_5 = points[:, 2] > 0.6
    mask = mask_1 | mask_2 | mask_3 | mask_4 | mask_5
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 0] < 0
    mask_2 = points[:, 0] > 0.3
    mask_3 = points[:, 1] < 0.9
    mask_4 = points[:, 1] > 1.3
    mask = mask_1 | mask_2 | mask_3 | mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 0] < -0.5
    mask_2 = points[:, 0] > 0.5
    mask_3 = points[:, 1] < -3.1
    mask_4 = points[:, 1] > -2.5
    mask_5 = points[:, 2] > 0.6
    mask = mask_1 | mask_2 | mask_3 | mask_4 | mask_5
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 0] < 0
    mask_2 = points[:, 0] > 0.3
    mask_3 = points[:, 1] < -1.3
    mask_4 = points[:, 1] > -0.9
    mask = mask_1 | mask_2 | mask_3 | mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "storage.pcd"), point_cloud)


def get_classroom_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "classroom.obj"), enable_post_processing=True
    )

    mesh.scale(0.002, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))

    center = mesh.get_center()
    mesh.translate((-center[0], -center[1], 3))

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh, coordinate])
    num_points = 500000
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 2] > 0
    mask = mask_1
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "classroom.pcd"), point_cloud)


def get_gym_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "gym.obj"), enable_post_processing=True
    )

    mesh.scale(0.012, center=(0, 0, 0))

    center = mesh.get_center()
    mesh.translate((-center[0], -center[1], -129.66))

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh, coordinate])
    num_points = 500000
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 2] > 0
    mask = mask_1
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "gym.pcd"), point_cloud)


def get_block_mc_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "block_mc.obj"), enable_post_processing=True
    )

    mesh.scale(10, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate((0, 0, -0.8))

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh, coordinate])
    num_points = 1000000
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    points = np.asarray(point_cloud.points)
    mask_1 = abs(points[:, 0]) < 4.7
    mask_2 = abs(points[:, 1]) < 11
    mask_3 = points[:, 2] < 1.4
    mask_4 = points[:, 2] > 0.0
    mask = mask_1 & mask_2 & mask_3 & mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "block_mc.pcd"), point_cloud)


def get_corridor_mc_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "corridor_mc.obj"), enable_post_processing=True
    )

    mesh.scale(10, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate((0, 0, -2.0))

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh, coordinate])
    num_points = 1000000
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    points = np.asarray(point_cloud.points)
    mask_1 = abs(points[:, 0]) < 10.5
    mask_2 = abs(points[:, 1]) < 15
    mask_3 = points[:, 2] < 1.2
    mask_4 = points[:, 2] > 0.0
    mask = mask_1 & mask_2 & mask_3 & mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 0] < -8
    mask_2 = points[:, 1] < 3.1
    mask_3 = points[:, 0] > -4
    mask_4 = points[:, 1] > 4.0
    mask = mask_1 | mask_2 | mask_3 | mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 0] < -5
    mask_2 = points[:, 1] < 5
    mask_3 = points[:, 0] > -3.5
    mask_4 = points[:, 1] > 7
    mask = mask_1 | mask_2 | mask_3 | mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "corridor_mc.pcd"), point_cloud)


def get_hui_mc_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "hui_mc.obj"), enable_post_processing=True
    )

    mesh.scale(10, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate((0, 0, -4.9))

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh, coordinate])
    num_points = 1000000
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    points = np.asarray(point_cloud.points)
    mask_1 = abs(points[:, 0]) < 5.5
    mask_2 = abs(points[:, 1]) < 2.5
    mask_3 = points[:, 2] < 1.4
    mask_4 = points[:, 2] > 0.0
    mask = mask_1 & mask_2 & mask_3 & mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "hui_mc.pcd"), point_cloud)


def get_villiage_mc_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "villiage_mc.obj"), enable_post_processing=True
    )

    mesh.scale(10, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate((0, 0, -2.2))

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh, coordinate])
    num_points = 1000000
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    points = np.asarray(point_cloud.points)
    mask_1 = abs(points[:, 0]) < 5.6
    mask_2 = abs(points[:, 1]) < 9.8
    mask_3 = points[:, 2] < 2.8
    mask_4 = points[:, 2] > 0.0
    mask = mask_1 & mask_2 & mask_3 & mask_4
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "villiage_mc.pcd"), point_cloud)


def get_Forest_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "Forest.obj"), enable_post_processing=True
    )

    mesh.scale(2.0, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate((0, 0, 0.25))

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh, coordinate])
    num_points = 500000
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    points = np.asarray(point_cloud.points)
    mask_1 = points[:, 2] > 0
    mask = mask_1
    point_cloud.points = o3d.utility.Vector3dVector(points[mask])

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "Forest.pcd"), point_cloud)
# others


def get_real_shape():
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(source_obj, "real.obj"), enable_post_processing=True
    )

    mesh.scale(0.001, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate((-0.55, -3.3734, -0.1))

    merged_mesh = copy.deepcopy(mesh)
    print(f"Center of mesh: {mesh.get_center()}")

    merged_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([merged_mesh, coordinate])
    num_points = 500000
    point_cloud = merged_mesh.sample_points_uniformly(number_of_points=num_points)

    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(os.path.join(source, "real.pcd"), point_cloud)


get_flower_shape()
