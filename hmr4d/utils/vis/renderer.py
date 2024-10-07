import cv2
import torch
import numpy as np

from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.transforms import axis_angle_to_matrix

from .renderer_tools import get_colors, checkerboard_geometry


colors_str_map = {
    "gray": [0.8, 0.8, 0.8],
    "green": [39, 194, 128],
}


def overlay_image_onto_background(image, mask, bbox, background):
    """
    将渲染的图像叠加到背景图像上。

    参数:
    - image: 渲染的图像
    - mask: 图像的遮罩
    - bbox: 边界框
    - background: 背景图像

    返回:
    - 叠加后的图像
    """
    # 将输入转换为numpy数组
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # 复制背景图像
    out_image = background.copy()
    # 获取边界框坐标
    bbox = bbox[0].int().cpu().numpy().copy()
    # 提取感兴趣区域(ROI)
    roi_image = out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    # 在遮罩区域应用渲染的图像
    roi_image[mask] = image[mask]
    # 将处理后的ROI放回原图
    out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype

    K = torch.zeros((K_org.shape[0], 4, 4)).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1

    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = right - left
    height = bottom - top

    new_left = torch.clamp(cx - width / 2 * scaleFactor, min=0, max=img_w - 1)
    new_right = torch.clamp(cx + width / 2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h - 1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = torch.stack((new_left.detach(), new_top.detach(), new_right.detach(), new_bottom.detach())).int().float().T

    return bbox


class Renderer:
    def __init__(self, width, height, focal_length=None, device="cuda", faces=None, K=None, bin_size=None):
        """
        初始化渲染器。

        参数:
        - width, height: 输出图像的宽度和高度
        - focal_length: 相机焦距(可选)
        - device: 计算设备
        - faces: 网格面部信息
        - K: 相机内参矩阵(可选)
        - bin_size: 光栅化的bin大小
        """
        self.width = width
        self.height = height
        self.bin_size = bin_size
        assert (focal_length is not None) ^ (K is not None), "focal_length和K必须提供其中之一"

        self.device = device
        if faces is not None:
            if isinstance(faces, np.ndarray):
                faces = torch.from_numpy((faces).astype("int"))
            self.faces = faces.unsqueeze(0).to(self.device)

        self.initialize_camera_params(focal_length, K)
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer()

    def create_renderer(self):
        """
        创建PyTorch3D渲染器。

        PyTorch3D渲染器由两个主要组件组成:
        1. Rasterizer (光栅化器): 将3D网格投影到2D图像平面。
        2. Shader (着色器): 计算每个像素的颜色。

        注意:
        - MeshRenderer 组合了 Rasterizer 和 Shader。
        - RasterizationSettings 控制光栅化过程的参数。
        - SoftPhongShader 使用 Phong 照明模型进行着色。
        """
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    # image_size: 输出图像的大小 (H, W)
                    image_size=self.image_sizes[0],
                    # blur_radius: 用于平滑边缘的模糊半径
                    blur_radius=1e-5,
                    # bin_size: 用于加速光栅化的空间划分大小
                    bin_size=self.bin_size
                ),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            ),
        )

    def create_camera(self, R=None, T=None):
        """
        创建PyTorch3D的PerspectiveCameras对象。

        PerspectiveCameras 用于定义相机的内参和外参,用于3D到2D的投影。

        参数:
        - R: 旋转矩阵, 形状为 (1, 3, 3) 或 None
        - T: 平移向量, 形状为 (1, 3) 或 None

        返回:
        - PerspectiveCameras 对象

        注意:
        - PerspectiveCameras 的主要参数:
          1. R: 旋转矩阵, 形状为 (N, 3, 3) 或 (N, 4, 4)
          2. T: 平移向量, 形状为 (N, 3)
          3. K: 相机内参矩阵, 形状为 (N, 4, 4) 或 (N, 3, 3)
          4. image_size: 图像尺寸, 形状为 ((N, 2))
          5. in_ndc: 是否使用标准化设备坐标 (NDC)

        - N 是批次大小,在这里通常为 1

        使用方法:
        1. 创建相机对象:
           cameras = PerspectiveCameras(R=R, T=T, K=K, image_size=image_size, device=device)
        2. 使用相机进行投影:
           projected_points = cameras.transform_points(points_3d)
        3. 在渲染过程中使用:
           renderer(meshes, cameras=cameras, ...)
        """
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device,
            R=self.R.mT,  # PyTorch3D期望R是相机到世界的变换,所以我们需要转置
            T=self.T,
            K=self.K_full,  # 完整的4x4内参矩阵
            image_size=self.image_sizes,
            in_ndc=False  # 使用像素坐标而不是NDC
        )

    def initialize_camera_params(self, focal_length, K):
        """
        初始化相机参数。

        这个方法设置了相机的内参和外参,并创建了初始的相机对象。

        参数:
        - focal_length: 相机焦距 (如果没有提供K)
        - K: 相机内参矩阵 (如果提供)

        注意:
        - 外参 (R 和 T) 被初始化为单位旋转和零平移
        - 内参 (K) 可以直接提供,或者通过focal_length计算
        - self.cameras 是创建的PerspectiveCameras对象,用于后续渲染
        """
        # 初始化外参 (相机姿态)
        self.R = torch.diag(torch.tensor([1, 1, 1])).float().to(self.device).unsqueeze(0)  # (1, 3, 3)
        self.T = torch.tensor([0, 0, 0]).unsqueeze(0).float().to(self.device)  # (1, 3)

        # 初始化内参
        if K is not None:
            self.K = K.float().reshape(1, 3, 3).to(self.device)
        else:
            assert focal_length is not None, "focal_length or K should be provided" # "必须提供focal_length或K"
            self.K = (
                torch.tensor([[focal_length, 0, self.width / 2], 
                              [0, focal_length, self.height / 2], 
                              [0, 0, 1]])
                .float()
                .reshape(1, 3, 3)
                .to(self.device)
            )
        
        # 设置初始边界框为整个图像
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        
        # 更新内参和图像尺寸
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        
        # 创建相机对象
        self.cameras = self.create_camera()

    def set_intrinsic(self, K):
        """
        设置相机内参。

        参数:
        - K: 新的相机内参矩阵
        """
        self.K = K.reshape(1, 3, 3)

    def set_ground(self, length, center_x, center_z):
        """
        设置地面几何信息。

        参数:
        - length: 地面长度
        - center_x, center_z: 地面中心坐标
        """
        device = self.device
        length, center_x, center_z = map(float, (length, center_x, center_z))
        v, f, vc, fc = map(torch.from_numpy, checkerboard_geometry(length=length, c1=center_x, c2=center_z, up="y"))
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]

    def update_bbox(self, x3d, scale=2.0, mask=None):
        """Update bbox of cameras from the given 3d points
        根据给定的3D点更新相机的边界框。

        参数:
        - x3d: 输入的3D关键点或顶点 input 3D keypoints (or vertices), (num_frames, num_points, 3)
        - scale: 缩放因子
        - mask: 遮罩(可选)
        """
        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))

        if mask is not None:
            x2d = x2d[:, ~mask]

        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(self):
        """
        重置边界框到全图大小。
        """
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def render_mesh(self, vertices, background=None, colors=[0.8, 0.8, 0.8], VI=50):
        """
        渲染3D网格。

        这个方法使用之前创建的self.cameras (PerspectiveCameras对象) 进行渲染。

        参数:
        - vertices: 网格顶点, 形状为 (V, 3), V 是顶点数量
        - background: 背景图像 (可选), 形状为 (H, W, 3)
        - colors: 网格颜色, 可以是单一颜色 [R, G, B] 或每个顶点的颜色 (V, 3)
        - VI: 顶点采样间隔

        返回:
        - 渲染后的图像, 形状为 (H, W, 3)

        注意:
        - self.renderer(...) 的调用过程:
          1. 输入:
             - mesh: Meshes 对象, 包含顶点 (1, V, 3) 和面 (1, F, 3)
             - materials: Materials 对象, 定义材质属性
             - cameras: Cameras 对象, 定义相机参数
             - lights: Lights 对象, 定义光照条件
          2. 光栅化:
             - 将3D网格投影到2D图像平面
             - 输出: (1, H, W, K) 张量, K包含深度和重心坐标
          3. 着色:
             - 使用Phong照明模型计算每个像素的颜色
             - 输出: (1, H, W, 4) 张量, 最后一个通道是alpha值
          4. 最终输出:
             - (1, H, W, 4) 张量, RGB颜色和alpha值
        """
        # 更新边界框，每VI个顶点采样一次，缩放比例为1.2
        self.update_bbox(vertices[::VI], scale=1.2)
        # 增加批次维度
        vertices = vertices.unsqueeze(0)

        # 处理顶点颜色
        if isinstance(colors, torch.Tensor):
            # per-vertex color
            # 如果提供了每个顶点的颜色
            verts_features = colors.to(device=vertices.device, dtype=vertices.dtype)
            colors = [0.8, 0.8, 0.8]
        else:
            # 如果提供了单一颜色
            if colors[0] > 1:
                colors = [c / 255.0 for c in colors]
            verts_features = torch.tensor(colors).reshape(1, 1, 3).to(device=vertices.device, dtype=vertices.dtype)
            verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        
        # 创建顶点纹理
        textures = TexturesVertex(verts_features=verts_features)

        # 创建网格
        mesh = Meshes(
            verts=vertices,
            faces=self.faces,
            textures=textures,
        )

        # 创建材质
        materials = Materials(device=self.device, specular_color=(colors,), shininess=0)

        # 使用self.cameras进行渲染
        results = torch.flip(self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights), [1, 2])
        # results 形状: (1, H, W, 4), 包含RGB颜色和alpha值
        # torch.flip 用于翻转图像, 因为PyTorch3D的输出是上下颠倒的

        # 提取渲染结果中的图像和遮罩
        image = results[0, ..., :3] * 255  # (H, W, 3), 范围 0-255
        mask = results[0, ..., -1] > 1e-3  # (H, W), 布尔遮罩

        # 如果没有提供背景，创建一个白色背景
        if background is None:
            background = np.ones((self.height, self.width, 3)).astype(np.uint8) * 255

        # 将渲染的图像叠加到背景上
        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
        # 重置边界框
        self.reset_bbox()
        return image

    def render_with_ground(self, verts, colors, cameras, lights, faces=None):
        """
        渲染带有地面的3D网格。

        参数:
        - verts: 网格顶点, (N, V, 3) potential multiple people
        - colors: 顶点颜色, (N, 3) 或 (N, V, 3)
        - cameras: 相机对象
        - lights: 光源对象
        - faces: 面部信息(可选), (N, F, 3) optional, otherwise self.faces is used will be used

        返回:
        - 渲染后的图像
        """
        # Sanity check of input verts, colors and faces: (B, V, 3), (B, F, 3), (B, V, 3)
        # 检查输入的形状
        N, V, _ = verts.shape
        if faces is None:
            faces = self.faces.clone().expand(N, -1, -1)
        else:
            assert len(faces.shape) == 3, "faces should have shape of (N, F, 3)" # "faces应该有(N, F, 3)的形状"

        assert len(colors.shape) in [2, 3]
        if len(colors.shape) == 2:
            assert len(colors) == N, "colors of shape 2 should be (N, 3)" # "colors的形状为2时应该是(N, 3)"
            colors = colors[:, None]
        colors = colors.expand(N, V, -1)[..., :3]

        # (V, 3), (F, 3), (V, 3)
        # 获取地面几何信息
        gv, gf, gc = self.ground_geometry
        verts = list(torch.unbind(verts, dim=0)) + [gv]
        faces = list(torch.unbind(faces, dim=0)) + [gf]
        colors = list(torch.unbind(colors, dim=0)) + [gc[..., :3]]
        mesh = create_meshes(verts, faces, colors)

        materials = Materials(device=self.device, shininess=0)

        results = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

        return image


def create_meshes(verts, faces, colors):
    """
    创建多个网格的场景。

    参数:
    - verts: 顶点列表, [(V, 3), ...]
    - faces: 面部信息列表, [(F, 3), ...]
    - colors: 颜色列表, [(V, 3), ...]

    返回:
    - 合并后的网格场景
    """
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)


def get_global_cameras(verts, device="cuda", distance=5, position=(-5.0, 5.0, 0.0)):
    """This always put object at the center of view
    获取全局相机参数。

    参数:
    - verts: 顶点坐标
    - device: 计算设备
    - distance: 相机距离
    - position: 相机初始位置

    返回:
    - rotation: 相机旋转矩阵
    - translation: 相机平移向量
    - lights: 光源对象
    """
    positions = torch.tensor([position]).repeat(len(verts), 1)
    targets = verts.mean(1)

    directions = targets - positions
    directions = directions / torch.norm(directions, dim=-1).unsqueeze(-1) * distance
    positions = targets - directions

    rotation = look_at_rotation(positions, targets).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    lights = PointLights(device=device, location=[position])
    return rotation, translation, lights


def get_global_cameras_static(
    verts, beta=4.0, cam_height_degree=30, target_center_height=1.0, use_long_axis=False, vec_rot=45, device="cuda"
):
    """
    获取静态全局相机参数。

    参数:
    - verts: 顶点坐标
    - beta: 缩放因子
    - cam_height_degree: 相机高度角度
    - target_center_height: 目标中心高度
    - use_long_axis: 是否使用长轴
    - vec_rot: 向量旋转角度
    - device: 计算设备

    返回:
    - rotation: 相机旋转矩阵
    - translation: 相机平移向量
    - lights: 光源对象
    """
    L, V, _ = verts.shape

    # Compute target trajectory, denote as center + scale
    # 计算目标轨迹
    targets = verts.mean(1)  # (L, 3)
    targets[:, 1] = 0  # project to xz-plane 投影到xz平面
    target_center = targets.mean(0)  # (3,)
    target_scale, target_idx = torch.norm(targets - target_center, dim=-1).max(0)

    # a 45 degree vec from longest axis 计算45度向量
    if use_long_axis:
        long_vec = targets[target_idx] - target_center  # (x, 0, z)
        long_vec = long_vec / torch.norm(long_vec)
        R = axis_angle_to_matrix(torch.tensor([0, np.pi / 4, 0])).to(long_vec)
        vec = R @ long_vec
    else:
        vec_rad = vec_rot / 180 * np.pi
        vec = torch.tensor([np.sin(vec_rad), 0, np.cos(vec_rad)]).float()
        vec = vec / torch.norm(vec)

    # Compute camera position (center + scale * vec * beta) + y=4 计算相机位置
    target_scale = max(target_scale, 1.0) * beta
    position = target_center + vec * target_scale
    position[1] = target_scale * np.tan(np.pi * cam_height_degree / 180) + target_center_height

    # Compute camera rotation and translation 计算相机旋转和平移
    positions = position.unsqueeze(0).repeat(L, 1)
    target_centers = target_center.unsqueeze(0).repeat(L, 1)
    target_centers[:, 1] = target_center_height
    rotation = look_at_rotation(positions, target_centers).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    lights = PointLights(device=device, location=[position.tolist()])
    return rotation, translation, lights


def get_ground_params_from_points(root_points, vert_points):
    """xz-plane is the ground plane
    从点云获取地面参数。

    参数:
    - root_points: 根点坐标, (L, 3), to decide center
    - vert_points: 顶点坐标, (L, V, 3), to decide scale

    返回:
    - scale: 缩放因子
    - cx, cz: 中心坐标
    """
    root_max = root_points.max(0)[0]  # (3,)
    root_min = root_points.min(0)[0]  # (3,)
    cx, _, cz = (root_max + root_min) / 2.0

    vert_max = vert_points.reshape(-1, 3).max(0)[0]  # (L, 3)
    vert_min = vert_points.reshape(-1, 3).min(0)[0]  # (L, 3)
    scale = (vert_max - vert_min)[[0, 2]].max()
    return float(scale), float(cx), float(cz)