import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
print('PyTorch 已安装' if torch.__version__ else 'PyTorch 未安装')
print('CUDA 可用' if torch.cuda.is_available() else 'CUDA 不可用')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

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
    """
    从物体的变换生成相机的变换
    """
    camera_transformations = []
    for object_transform in object_transformations:
        # 相机的变换是物体变换的逆
        camera_transform = np.linalg.inv(object_transform)
        camera_transformations.append(camera_transform)
    return camera_transformations

# 主要参数
num_steps = 72
rotation_axis = np.array([0, 1, 0])  # 绕y轴旋转
rotation_angle_step = np.radians(5)
initial_position = np.array([0, 0, 2])

# 生成变换数据
object_transformations = generate_transformations(num_steps, rotation_axis, rotation_angle_step, initial_position)
camera_transformations = generate_camera_transformations(object_transformations)

# 创建物体A（立方体）的顶点
cube_points = np.array([
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5]
])

# 定义立方体的边
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
    (0, 4), (1, 5), (2, 6), (3, 7)   # 连接底面和顶面的边
]

# 创建相机模型（简化为一个四棱锥，朝向z轴正方向）
camera_points = np.array([
    [0, 0, 0],      # 相机原点
    [-0.2, -0.2, 0.5],  # 前面的四个点
    [0.2, -0.2, 0.5],
    [0.2, 0.2, 0.5],
    [-0.2, 0.2, 0.5]
])

# 定义相机模型的边
camera_edges = [
    (0, 1), (0, 2), (0, 3), (0, 4),  # 从原点到前面四个点的边
    (1, 2), (2, 3), (3, 4), (4, 1)   # 前面的四边形
]

# 设置图形
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# 用于存储相机位置的列表
camera_positions = []

def init():
    for ax in [ax1, ax2]:
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_zlim(-2.5, 2.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # 设置相等的轴比例
        ax.set_box_aspect((1, 1, 1))
    ax1.set_title("Scene 1: Object A Rotation")
    ax2.set_title("Scene 2: Camera Rotation")
    return []

def animate(i):
    ax1.clear()
    ax2.clear()
    init()

    # 场景一：物体A旋转
    transformed_cube = apply_transform(cube_points, object_transformations[i])
    ax1.scatter(transformed_cube[:, 0], transformed_cube[:, 1], transformed_cube[:, 2])
    for edge in cube_edges:
        ax1.plot(transformed_cube[edge, 0], transformed_cube[edge, 1], transformed_cube[edge, 2], 'b')
    
    # 绘制固定位置的相机
    fixed_camera = apply_transform(camera_points, np.eye(4))
    ax1.scatter(fixed_camera[:, 0], fixed_camera[:, 1], fixed_camera[:, 2], color='red')
    for edge in camera_edges:
        ax1.plot(fixed_camera[edge, 0], fixed_camera[edge, 1], fixed_camera[edge, 2], 'r')

    # 场景二：相机旋转
    transformed_camera = apply_transform(camera_points, camera_transformations[i])
    ax2.scatter(transformed_camera[:, 0], transformed_camera[:, 1], transformed_camera[:, 2], color='red')
    for edge in camera_edges:
        ax2.plot(transformed_camera[edge, 0], transformed_camera[edge, 1], transformed_camera[edge, 2], 'r')
    
    # 保存相机位置并绘制轨迹
    camera_position = transformed_camera[0]  # 相机的原点
    camera_positions.append(camera_position)
    if len(camera_positions) > 1:
        camera_trajectory = np.array(camera_positions)
        ax2.plot(camera_trajectory[:, 0], camera_trajectory[:, 1], camera_trajectory[:, 2], 'r-', linewidth=0.5, alpha=0.5)
    
    # 绘制固定的物体A
    ax2.scatter(cube_points[:, 0], cube_points[:, 1], cube_points[:, 2])
    for edge in cube_edges:
        ax2.plot(cube_points[edge, 0], cube_points[edge, 1], cube_points[edge, 2], 'b')

    # 添加坐标轴
    for ax in [ax1, ax2]:
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=1, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=1, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=1, arrow_length_ratio=0.1)
        ax.text(1.1, 0, 0, 'X', color='r')
        ax.text(0, 1.1, 0, 'Y', color='g')
        ax.text(0, 0, 1.1, 'Z', color='b')

    return []

# 创建动画
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_steps, interval=100, blit=True)

plt.tight_layout()
plt.show()