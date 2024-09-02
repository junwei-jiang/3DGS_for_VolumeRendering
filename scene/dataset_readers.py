#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import math
import re
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def read_vol_cam_extr_text(file_path):
    # Regular expression pattern to capture ID, Position, Look At, Up, and optionally View Matrix
    camera_info_pattern = r'ID: (\d+).?Position: (([^)]+)).?Look At: (([^)]+)).?Up: (([^)]+))(?:.?View Matrix: (([^)]+)))?'
    camera_info_pattern = r'ID:\s(\d+).?Position:\s(([^)]+)).?Look At:\s(([^)]+)).?Up:\s(([^)]+))(?:.?View Matrix:\s*(([^)]+)))?'
    camera_info_pattern = r'ID:\s(\d+)\s*Position:\s\(([^)]+)\)\s*Look At:\s\(([^)]+)\)\s*Up:\s\(([^)]+)\)\s*View Matrix:\s\(([^)]+)\)'

    cam_extrinsics = []
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()
        for match in re.finditer(camera_info_pattern, file_contents, re.DOTALL):
            id_str, position_str, look_at_str, up_str, view_matrix_str = match.groups()
            camera_id = id_str
            position = tuple(map(float, position_str.split(', ')))
            look_at = tuple(map(float, look_at_str.split(', ')))
            up = tuple(map(float, up_str.split(', ')))

            camera_info = {'ID': camera_id, 'Position': position, 'Look At': look_at, 'Up': up}
            if view_matrix_str:
                view_matrix = tuple(map(float, view_matrix_str.split(', ')))
                camera_info['View Matrix'] = view_matrix

            cam_extrinsics.append(camera_info)

    return cam_extrinsics
def read_vol_cam_intr_text(file_path):
    # Regular expression pattern to capture Width, Height, and FoVy
    camera_info_pattern = r'Width: (\d+).?Height: (\d+).?FoVy: ([-\d.]+)'
    cam_intrinsics = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()
        match = re.search(camera_info_pattern, file_contents, re.DOTALL)
        if match:
            width_str, height_str, fovy_str = match.groups()
            cam_intrinsics['Width'] = int(width_str)
            cam_intrinsics['Height'] = int(height_str)
            cam_intrinsics['FovY'] = float(fovy_str)

    return cam_intrinsics

def readCTMeta(meta_folder): # For example: volume_shape = (256, 256, 161)
    # Walk through the directory
    for root, dirs, files in os.walk(meta_folder):
        for file in files:
            # Check if the file ends with .dat
            if file.endswith('.dat'):
                # Add the file path to the list
                file_path = os.path.join(root, file)
                break

    metadata = {
        "ObjectFileName": None,
        "Resolution": None,
        "SliceThickness": None,
        "Format": None
    }

    with open(file_path, 'r') as file:
        for line in file:
            if "ObjectFileName" in line:
                metadata["ObjectFileName"] = line.split(':')[1].strip()
            elif "Resolution:" in line:
                # Convert the resolution to integers
                metadata["Resolution"] = [int(x) for x in line.split(':')[1].strip().split()]
            elif "SliceThickness:" in line:
                # Convert the slice thickness to floats
                metadata["SliceThickness"] = [float(x) for x in line.split(':')[1].strip().split()]
            elif "Format:" in line:
                metadata["Format"] = line.split(':')[1].strip()

    return metadata

def initialCloudPcd(shape, divs ):
    divs_x, divs_y, divs_z = divs # gaussian_num = divs_x * divs_y * divs_z
    print("initialize uniformly!\n")
    y, z, x = shape
    x_centers = np.linspace(-1 * x / 2 , x / 2, divs_x)
    y_centers = np.linspace(-1 * y / 2 , y / 2, divs_y)
    z_centers = np.linspace(-1 * z / 2 , z / 2, divs_z)
    # Create a grid of center coordinates
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    # initial colors
    colors = np.ones_like(positions) * 255.0 # initial color (255.0, 255.0, 255.0)
    normals = np.zeros_like(positions) # initial normal (0.0, 0.0, 0.0), we dont need them actually
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def readVolumeCameras(cam_extrinsics, cam_intrinsics, images_folder, return_cam_mat:bool=False):
    cam_infos = []
    for idx, cam_extrinsic in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        height = cam_intrinsics['Height']
        width = cam_intrinsics['Width']

        uid = 0

        look_at = np.array(cam_extrinsic['Look At'])
        at = np.array([ look_at[2], look_at[1], look_at[0]])
        cam_center = np.array(cam_extrinsic['Position'])
        eye = np.array([  cam_center[2], cam_center[1], cam_center[0]])
        up_dir = np.array(cam_extrinsic['Up'])
        up = np.array([ -1*up_dir[2],-1* up_dir[1], -1*up_dir[0]])

        zaxis = at - eye
        norm = np.linalg.norm(zaxis)
        zaxis = zaxis / norm

        xaxis = np.cross(zaxis, up)
        norm = np.linalg.norm(xaxis)
        xaxis = xaxis / norm

        yaxis = np.cross(xaxis, zaxis)
        C2W = np.array(
        [[xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
        [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
        [zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis, eye)],
        [0,0,0,1]]
        )
        R = C2W[:3, :3].transpose() # Transpose of the upper left 3x3 matrix
        T = C2W[:3, 3]

        image_id = cam_extrinsic['ID']

        FovY = cam_intrinsics['FovY']
        # Convert fovY to radians for the math functions
        fovY_radians = math.radians(FovY)
        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Calculate fovX using the formula
        fovX_radians = 2 * math.atan(math.tan(fovY_radians / 2) * aspect_ratio)
        image_path = os.path.join(images_folder, 'img_'+image_id+'.png')

        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=fovY_radians, FovX=fovX_radians, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readVolumeCameras2(cam_datas_file, images_folder, return_cam_mat:bool=False):
    cam_infos = []

    with open(cam_datas_file, 'r') as file:
        cam_datas = json.load(file)

    for idx, cam_data in enumerate(cam_datas):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_datas)))
        sys.stdout.flush()

        uid = 0

        c2w_matrix = np.array(cam_data['extrinsics']['c2w_matrix'])
        width = cam_data['width']
        height = cam_data['height']
        fovx = cam_data['fovx']
        fovy = cam_data['fovy']
        img_id = cam_data['img_id']

        R = c2w_matrix[:3, :3].transpose()  # Transpose of the upper left 3x3 matrix
        T = c2w_matrix[:3, 3]

        fovY_radians = math.radians(fovy)
        fovX_radians = math.radians(fovx)

        image_path = os.path.join(images_folder, str(img_id) + '.png')

        image_name = str(img_id)
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=img_id, R=R, T=T, FovY=fovY_radians, FovX=fovX_radians, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)

        if idx==1:
            print(cam_info)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readCustomDataInfo(path, eval, gaussian_num=(30,30,30), llffhold=8):
    """
    path: the source path, where the rendered images
    initial pcd and saved in ply file
    read camera info for each image
    """

    cameras_info_file = os.path.join(path, "images.txt") #TODO: this will be changed!
    camera_intrinsic_file = os.path.join(path, "camera.txt")
    cam_extrinsics = read_vol_cam_extr_text(cameras_info_file)
    cam_intrinsic = read_vol_cam_intr_text(camera_intrinsic_file)
    reading_dir = "images"
    ct_meta_dir = "cloud"
    images_folder = os.path.join(path, reading_dir)
    meta_folder = os.path.join(path, ct_meta_dir)

    # for each image, generate CameraInfo List,
    cam_infos_unsorted = readVolumeCameras(cam_extrinsics, cam_intrinsic, images_folder)

    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    metadata = readCTMeta(meta_folder) #volume_shape = (256, 256, 161)
    volume_shape = metadata["Resolution"]
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []


    nerf_normalization = getNerfppNorm(train_cam_infos)


    ply_path = os.path.join(path, "points3D.ply") # the path of the initial point cloud: grid gaussians

    pcd = initialCloudPcd(volume_shape, gaussian_num)

    storePly(ply_path, pcd.points, pcd.colors)
    pcd = fetchPly(ply_path) #! intial point cloud from the SfM (80861 points) or initial grid pcd

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# def readCustomDataInfo2(path, eval, gaussian_num=(30,30,30), llffhold=8):
#     """
#     path: the source path, where the rendered images
#     initial pcd and saved in ply file
#     read camera info for each image
#     """
#     cameras_data_file = os.path.join(path, "cameras.json")
#     reading_dir = "images"
#     images_folder = os.path.join(path, reading_dir)
#     ct_meta_dir = "meta"
#     meta_folder = os.path.join(path, ct_meta_dir)
#     metadata = readCTMeta(meta_folder)  # volume_shape = (256, 256, 161)
#     volume_shape = metadata["Resolution"]
#
#     # for each image, generate CameraInfo List,
#     cam_infos_unsorted = readVolumeCameras2(cameras_data_file, images_folder)
#
#     cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
#
#     if eval:
#         train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
#         test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []
#
#     nerf_normalization = getNerfppNorm(train_cam_infos)
#
#     ply_path = os.path.join(path, "points3D.ply")  # the path of the initial point cloud: grid gaussians
#     pcd = initialCloudPcd(volume_shape, gaussian_num)
#
#     storePly(ply_path, pcd.points, pcd.colors)
#     pcd = fetchPly(ply_path)  # ! intial point cloud from the SfM (80861 points) or initial grid pcd
#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
#     return scene_info

def readCamerasFromTransforms2(path, transformsfile, white_background, extension=".png", t = 'train'):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        cam_datas = json.load(json_file)


        for idx, cam_data in enumerate(cam_datas):
            if t=='train':
                cam_name = os.path.join(path, "./train/image" + str(idx)+ extension)
            else:
                cam_name = os.path.join(path, "./test/image" + str(idx) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(cam_data["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovx = cam_data["camera_angle_x"]
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos

def readCustomDataInfo2(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms2(path, "transforms_train.json", white_background, extension, 'train')
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms2(path, "transforms_test.json", white_background, extension, 'test')

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Custom": readCustomDataInfo,
    "Custom2": readCustomDataInfo2
}