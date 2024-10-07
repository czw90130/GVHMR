import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 保留原始函数
def create_rotation_matrix(axis, angle):
    """创建绕指定轴旋转的旋转矩阵"""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
    ])

def create_transformation_matrix(rotation_matrix, translation_vector):
    """创建4x4齐次变换矩阵"""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation_vector
    return matrix

def generate_transformations(num_steps, rotation_axis, rotation_angle_step, initial_position):
    """生成变换数据"""
    transformations = []
    for i in range(num_steps):
        rotation_matrix = create_rotation_matrix(rotation_axis, rotation_angle_step * i)
        transformation = create_transformation_matrix(rotation_matrix, initial_position)
        transformations.append(transformation)
    return transformations

def apply_transform(points, transform):
    """应用变换到点集"""
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = np.dot(homogeneous_points, transform.T)
    return transformed_points[:, :3]

def generate_camera_transformations(object_transformations):
    """从物体的变换生成相机的变换"""
    camera_transformations = []
    for object_transform in object_transformations:
        camera_transform = np.linalg.inv(object_transform)
        camera_transformations.append(camera_transform)
    return camera_transformations

# 创建立方体顶点和面
def create_cube():
    vertices = torch.tensor([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ], dtype=torch.float32, device=device)
    
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # 前面
        [4, 5, 6], [4, 6, 7],  # 后面
        [0, 4, 7], [0, 7, 3],  # 左面
        [1, 5, 6], [1, 6, 2],  # 右面
        [3, 2, 6], [3, 6, 7],  # 上面
        [0, 1, 5], [0, 5, 4]   # 下面
    ], dtype=torch.long, device=device)
    
    return vertices, faces

# 创建渲染器
def create_renderer(image_size=512, device=device):
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1,
    )
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=None, lights=None)
    )
    
    return renderer

# 主函数
def main():
    # 参数设置
    image_size = 512
    num_steps = 72
    rotation_axis = np.array([0, 1, 0])  # 绕y轴旋转
    rotation_angle_step = np.radians(5)
    initial_position = np.array([0, 0, 2])

    # 生成变换数据
    object_transformations = generate_transformations(num_steps, rotation_axis, rotation_angle_step, initial_position)
    camera_transformations = generate_camera_transformations(object_transformations)

    # 创建立方体和渲染器
    vertices, faces = create_cube()
    renderer = create_renderer(image_size=image_size)

    # 设置材质
    vertex_colors = torch.ones_like(vertices) * 0.8  # 灰色
    textures = TexturesVertex(verts_features=vertex_colors.unsqueeze(0))

    # 创建网格
    meshes = Meshes(verts=[vertices], faces=[faces], textures=textures)

    # 设置光源
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # 准备图像数组
    images_scene1 = []
    images_scene2 = []

    # 渲染循环
    for i in range(num_steps):
        # 场景1：物体A旋转
        transform = torch.tensor(object_transformations[i], dtype=torch.float32, device=device)
        R = transform[:3, :3].unsqueeze(0)
        T = transform[:3, 3].unsqueeze(0)
        cameras = PerspectiveCameras(device=device, R=R, T=T)
        
        image_scene1 = renderer(meshes, cameras=cameras, lights=lights)
        images_scene1.append(image_scene1[0, ..., :3].cpu().numpy())

        # 场景2：相机旋转
        transform = torch.tensor(camera_transformations[i], dtype=torch.float32, device=device)
        R = transform[:3, :3].unsqueeze(0)
        T = transform[:3, 3].unsqueeze(0)
        cameras = PerspectiveCameras(device=device, R=R, T=T)
        
        image_scene2 = renderer(meshes, cameras=cameras, lights=lights)
        images_scene2.append(image_scene2[0, ..., :3].cpu().numpy())

    # 显示结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(images_scene1[0])
    ax1.set_title("Scene 1: Object A Rotation")
    ax1.axis('off')
    
    ax2.imshow(images_scene2[0])
    ax2.set_title("Scene 2: Camera Rotation")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

    # 创建动画（可选）
    # 这里您可以使用matplotlib的animation功能来创建动画
    # 或者将图像保存为视频文件

if __name__ == "__main__":
    main()