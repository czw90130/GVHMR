import torch
import numpy as np
from hmr4d.utils.geo_transform import project_p2d, convert_bbx_xys_to_lurb, cvt_to_bi01_p2d

# NOTE: 估计焦距
def estimate_focal_length(img_w, img_h):
    """
    估计图像的焦距
    参数:
        img_w: 图像宽度
        img_h: 图像高度
    返回:
        估计的焦距值
    """
    # Diagonal FOV = 2*arctan(0.5) * 180/pi = 53
    return (img_w**2 + img_h**2) ** 0.5  # 对角线FOV = 2*arctan(0.5) * 180/pi ≈ 53度

# NOTE: 估计内参
def estimate_K(img_w, img_h):
    """
    估计相机内参矩阵K
    参数:
        img_w: 图像宽度
        img_h: 图像高度
    返回:
        估计的3x3内参矩阵K
    """
    focal_length = estimate_focal_length(img_w, img_h)
    K = torch.eye(3).float()  # 创建3x3单位矩阵
    K[0, 0] = focal_length  # 设置fx
    K[1, 1] = focal_length  # 设置fy
    K[0, 2] = img_w / 2.0   # 设置cx (主点x坐标)
    K[1, 2] = img_h / 2.0   # 设置cy (主点y坐标)
    return K

def convert_K_to_K4(K):
    """
    将3x3内参矩阵K转换为4元素向量形式
    参数:
        K: 3x3内参矩阵
    返回:
        K4: 包含[fx, fy, cx, cy]的4元素向量
    """
    K4 = torch.stack([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]).float()
    return K4

def convert_f_to_K(focal_length, img_w, img_h):
    """
    根据焦距和图像尺寸创建内参矩阵K
    参数:
        focal_length: 焦距
        img_w: 图像宽度
        img_h: 图像高度
    返回:
        3x3内参矩阵K
    """
    K = torch.eye(3).float()
    K[0, 0] = focal_length  # 设置fx
    K[1, 1] = focal_length  # 设置fy
    K[0, 2] = img_w / 2.0   # 设置cx
    K[1, 2] = img_h / 2.0   # 设置cy
    return K

def resize_K(K, f=0.5):
    """
    调整内参矩阵K的尺寸
    参数:
        K: 原始内参矩阵
        f: 缩放因子
    返回:
        调整后的内参矩阵
    """
    K = K.clone() * f  # 克隆并缩放K
    K[..., 2, 2] = 1.0  # 保持K[2,2]为1
    return K

def create_camera_sensor(width=None, height=None, f_fullframe=None):
    """
    创建相机传感器参数
    参数:
        width: 图像宽度
        height: 图像高度
        f_fullframe: 全画幅相机的焦距
    返回:
        width: 图像宽度
        height: 图像高度
        K_fullimg: 内参矩阵
    """
    if width is None or height is None:
        # The 4:3 aspect ratio is widely adopted by image sensors in mobile phones.
        # 如果未指定宽高,随机选择常见的4:3宽高比
        if np.random.rand() < 0.5:
            width, height = 1200, 1600
        else:
            width, height = 1600, 1200

    # Sample FOV from common options:
    # 从常见选项中采样FOV:
    # 1. wide-angle lenses are common in mobile phones,
    # 1. 广角镜头在手机中很常见
    # 2. telephoto lenses has less perspective effect, which should makes it easy to learn
    # 2. 长焦镜头透视效果较小,可能更容易学习
    if f_fullframe is None:
        f_fullframe_options = [24, 26, 28, 30, 35, 40, 50, 60, 70]
        f_fullframe = np.random.choice(f_fullframe_options)

    # We use diag to map focal-length: https://www.nikonians.org/reviews/fov-tables
    # 使用对角线长度映射焦距: https://www.nikonians.org/reviews/fov-tables
    diag_fullframe = (24**2 + 36**2) ** 0.5  # 全画幅传感器对角线长度
    diag_img = (width**2 + height**2) ** 0.5  # 当前图像对角线长度
    focal_length = diag_img / diag_fullframe * f_fullframe  # 计算等效焦距

    K_fullimg = torch.eye(3)
    K_fullimg[0, 0] = focal_length
    K_fullimg[1, 1] = focal_length
    K_fullimg[0, 2] = width / 2
    K_fullimg[1, 2] = height / 2

    return width, height, K_fullimg

# ====== Compute cliffcam ===== #

def convert_xys_to_cliff_cam_wham(xys, res):
    """
    将像素坐标转换为CLIFF相机的标准化表示
    Args:
        xys: (N, 3) in pixel. Note s should not be touched by 200 像素坐标。注意s不应被200影响
        res: (2) e.g. [4112., 3008.]  (w,h) 图像分辨率,例如 [4112., 3008.] (宽,高)
    Returns:
        cliff_cam: (N, 3) normalized representation 标准化表示
    """
    def normalize_keypoints_to_image(x, res):
        """
        将关键点坐标标准化到图像空间
        Args:
            x: (N, 2) centers 中心点坐标
            res: (2) 图像分辨率 e.g. [4112., 3008.]
        Returns:
            x_normalized: (N, 2) 标准化后的坐标
        """
        res = res.to(x.device)
        scale = res.max(-1)[0].reshape(-1)
        mean = torch.stack([res[..., 0] / scale, res[..., 1] / scale], dim=-1).to(x.device)
        x = 2 * x / scale.reshape(*[1 for i in range(len(x.shape[1:]))]) - mean.reshape(
            *[1 for i in range(len(x.shape[1:-1]))], -1
        )
        return x

    centers = normalize_keypoints_to_image(xys[:, :2], res)  # (N, 2)
    scale = xys[:, 2:] / res.max()
    location = torch.cat((centers, scale), dim=-1)
    return location

def compute_bbox_info_bedlam(bbx_xys, K_fullimg):
    """impl as in BEDLAM
    计算BEDLAM中的边界框信息
    Args:
        bbx_xys: ((B), N, 3), in pixel space described by K_fullimg 像素空间中的边界框坐标
        K_fullimg: ((B), (N), 3, 3) 内参矩阵
    Returns:
        bbox_info: ((B), N, 3) 边界框信息
    """
    fl = K_fullimg[..., 0, 0].unsqueeze(-1)  # 焦距
    icx = K_fullimg[..., 0, 2]  # 主点x坐标
    icy = K_fullimg[..., 1, 2]  # 主点y坐标

    cx, cy, b = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    bbox_info = torch.stack([cx - icx, cy - icy, b], dim=-1)
    bbox_info = bbox_info / fl  # 归一化
    return bbox_info

# ====== Convert Prediction to Cam-t ===== #

def compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg):
    """
    计算完整相机空间中的平移向量
    参数:
        pred_cam: 预测的相机参数
        bbx_xys: 边界框坐标
        K_fullimg: 内参矩阵
    返回:
        cam_t: 相机空间中的平移向量
    """
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]

    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t

def get_a_pred_cam(transl, bbx_xys, K_fullimg):
    """Inverse operation of compute_transl_full_cam
    compute_transl_full_cam的逆操作
    参数:
        transl: 平移向量
        bbx_xys: 边界框坐标
        K_fullimg: 内参矩阵
    返回:
        gt_pred_cam: 预测的相机参数
    """
    assert transl.ndim == bbx_xys.ndim  # (*, L, 3)
    assert K_fullimg.ndim == (bbx_xys.ndim + 1)  # (*, L, 3, 3)
    f = K_fullimg[..., 0, 0]
    cx = K_fullimg[..., 0, 2]
    cy = K_fullimg[..., 1, 2]
    gt_s = 2 * f / (transl[..., 2] * bbx_xys[..., 2])  # (B, L)
    gt_x = transl[..., 0] - transl[..., 2] / f * (bbx_xys[..., 0] - cx)
    gt_y = transl[..., 1] - transl[..., 2] / f * (bbx_xys[..., 1] - cy)
    gt_pred_cam = torch.stack([gt_s, gt_x, gt_y], dim=-1)
    return gt_pred_cam

# ====== 3D to 2D ===== #

def project_to_bi01(points, bbx_xys, K_fullimg):
    """
    将3D点投影到2D并标准化到[-1,1]范围
    参数:
        points: (B, L, J, 3) 3D点坐标
        bbx_xys: (B, L, 3) 边界框坐标
        K_fullimg: (B, L, 3, 3) 内参矩阵
    返回:
        p2d_bi01: 标准化的2D投影点
    """
    # p2d = project_p2d(points, K_fullimg)
    p2d = perspective_projection(points, K_fullimg)
    bbx_lurb = convert_bbx_xys_to_lurb(bbx_xys)
    p2d_bi01 = cvt_to_bi01_p2d(p2d, bbx_lurb)
    return p2d_bi01

def perspective_projection(points, K):
    """
    透视投影
    参数:
        points: (B, L, J, 3) 3D点坐标
        K: (B, L, 3, 3) 内参矩阵
    返回:
        projected_points: 投影后的2D点坐标
    """
    projected_points = points / points[..., -1].unsqueeze(-1)
    projected_points = torch.einsum("...ij,...kj->...ki", K, projected_points.float())
    return projected_points[..., :-1]

# ====== 2D (bbx from j2d) ===== #

def normalize_kp2d(obs_kp2d, bbx_xys, clamp_scale_min=False):
    """
    标准化2D关键点
    Args:
        obs_kp2d: (B, L, J, 3) [x, y, c] 观察到的2D关键点
        bbx_xys: (B, L, 3) 边界框坐标
        clamp_scale_min: 是否限制最小缩放值
    Returns:
        obs: 标准化后的2D关键点 (B, L, J, 3)  [x, y, c]
    """
    obs_xy = obs_kp2d[..., :2]  # (B, L, J, 2) 提取xy坐标
    obs_conf = obs_kp2d[..., 2]  # (B, L, J) 提取置信度
    center = bbx_xys[..., :2] # 边界框中心
    scale = bbx_xys[..., [2]] # 边界框尺寸

    # Mark keypoints outside the bounding box as invisible
    # 将边界框外的关键点标记为不可见
    xy_max = center + scale / 2 # 边界框右下角
    xy_min = center - scale / 2 # 边界框左上角
    invisible_mask = (
        (obs_xy[..., 0] < xy_min[..., None, 0])
        + (obs_xy[..., 0] > xy_max[..., None, 0])
        + (obs_xy[..., 1] < xy_min[..., None, 1])
        + (obs_xy[..., 1] > xy_max[..., None, 1])
    )
    obs_conf = obs_conf * ~invisible_mask  # 更新置信度
    if clamp_scale_min:
        scale = scale.clamp(min=1e-5) # 限制最小缩放值
    normalized_obs_xy = 2 * (obs_xy - center.unsqueeze(-2)) / scale.unsqueeze(-2)  # 标准化坐标

    return torch.cat([normalized_obs_xy, obs_conf[..., None]], dim=-1)


def get_bbx_xys(i_j2d, bbx_ratio=[192, 256], do_augment=False, base_enlarge=1.2):
    """
    从2D关键点获取边界框坐标
    参数:
        i_j2d: (B, L, J, 3) [x,y,c] 2D关键点
        bbx_ratio: 边界框比例
        do_augment: 是否进行数据增强
        base_enlarge: 基础放大系数
    返回:
        边界框坐标 (B, L, 3)
    """
    # Center
    # 计算中心点
    min_x = i_j2d[..., 0].min(-1)[0]
    max_x = i_j2d[..., 0].max(-1)[0]
    min_y = i_j2d[..., 1].min(-1)[0]
    max_y = i_j2d[..., 1].max(-1)[0]
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Size
    # 计算尺寸
    h = max_y - min_y  # (B, L)
    w = max_x - min_x  # (B, L)

    if True:  # fit w and h into aspect-ratio 调整w和h以适应指定的宽高比
        aspect_ratio = bbx_ratio[0] / bbx_ratio[1]
        mask1 = w > aspect_ratio * h
        h[mask1] = w[mask1] / aspect_ratio
        mask2 = w < aspect_ratio * h
        w[mask2] = h[mask2] * aspect_ratio

    # apply a common factor to enlarge the bounding box 应用放大系数
    bbx_size = torch.max(h, w) * base_enlarge

    if do_augment:
        B, L = bbx_size.shape[:2]
        device = bbx_size.device
        if True:
            scaleFactor = torch.rand((B, L), device=device) * 0.3 + 1.05  # 1.05~1.35
            txFactor = torch.rand((B, L), device=device) * 1.6 - 0.8  # -0.8~0.8
            tyFactor = torch.rand((B, L), device=device) * 1.6 - 0.8  # -0.8~0.8
        else:
            scaleFactor = torch.rand((B, 1), device=device) * 0.3 + 1.05  # 1.05~1.35
            txFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8  # -0.8~0.8
            tyFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8  # -0.8~0.8

        raw_bbx_size = bbx_size / base_enlarge
        bbx_size = raw_bbx_size * scaleFactor
        center_x += raw_bbx_size / 2 * ((scaleFactor - 1) * txFactor)
        center_y += raw_bbx_size / 2 * ((scaleFactor - 1) * tyFactor)

    return torch.stack([center_x, center_y, bbx_size], dim=-1)


def safely_render_x3d_K(x3d, K_fullimg, thr):
    """
    安全地渲染3D点
    Args:
        x3d: (B, L, V, 3), 3D点坐标 should as least have a safe points (not examined here)
        K_fullimg: (B, L, 3, 3) 内参矩阵
    Returns:
        bbx_xys: (B, L, 3)
        i_x2d: (B, L, V, 2) 2D投影点
    """
    #  For each frame, update unsafe z (<thr) to safe z (max)
    # 对于每一帧,将不安全点的z坐标更新为安全点的z坐标
    x3d = x3d.clone()  # (B, L, V, 3)
    x3d_unsafe_mask = x3d[..., 2] < thr  # (B, L, V) 不安全点的掩码
    if (x3d_unsafe_mask).sum() > 0:
        x3d[..., 2][x3d_unsafe_mask] = thr # 将不安全点的z坐标设为阈值
        if False:
            from hmr4d.utils.wis3d_utils import make_wis3d

            wis3d = make_wis3d(name="debug-update-z")
            bs, ls, vs = torch.where(x3d_unsafe_mask)
            bs = torch.unique(bs)
            for b in bs:
                for f in range(x3d.size(1)):
                    wis3d.set_scene_id(f)
                    wis3d.add_point_cloud(x3d[b, f], name="unsafe")
                pass

    # renfer 投影
    i_x2d = perspective_projection(x3d, K_fullimg)  # (B, L, V, 2)
    return i_x2d


def get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2):
    """
    从xyxy格式的边界框获取xys格式的边界框
    Args:
        bbx_xyxy: (N, 4) [x1, y1, x2, y2]
        base_enlarge: 基础放大系数
    Returns:
        bbx_xys: (N, 3) [center_x, center_y, size]
    """

    i_p2d = torch.stack([bbx_xyxy[:, [0, 1]], bbx_xyxy[:, [2, 3]]], dim=1)  # (L, 2, 2)
    bbx_xys = get_bbx_xys(i_p2d[None], base_enlarge=base_enlarge)[0]
    return bbx_xys


def bbx_xyxy_from_x(p2d):
    """
    从2D点获取xyxy格式的边界框
    Args:
        p2d: (*, V, 2) - Tensor containing 2D points. 2D点坐标

    Returns:
        bbx_xyxy: (*, 4) - Bounding box coordinates in the format 边界框坐标 (xmin, ymin, xmax, ymax).
    """
    # Compute the minimum and maximum coordinates for the bounding box
    # 计算边界框的最小和最大坐标
    xy_min = p2d.min(dim=-2).values  # (*, 2)
    xy_max = p2d.max(dim=-2).values  # (*, 2)

    # Concatenate min and max coordinates to form the bounding box
    # 将最小和最大坐标连接起来形成边界框
    bbx_xyxy = torch.cat([xy_min, xy_max], dim=-1)  # (*, 4)

    return bbx_xyxy


def bbx_xyxy_from_masked_x(p2d, mask):
    """
    从带掩码的2D点获取xyxy格式的边界框
    Args:
        p2d: (*, V, 2) - Tensor containing 2D points. 2D点坐标
        mask: (*, V) - Boolean tensor indicating valid points. 布尔掩码,指示有效点

    Returns:
        bbx_xyxy: (*, 4) - Bounding box coordinates in the format 边界框坐标 (xmin, ymin, xmax, ymax).
    """
    # Ensure the shapes of p2d and mask are compatible
    # 确保p2d和mask的形状兼容
    assert p2d.shape[:-1] == mask.shape, "The shape of p2d and mask are not compatible." # 确保p2d和mask的形状兼容

    # Flatten the input tensors for batch processing
    # 展平输入张量以进行批量处理
    p2d_flat = p2d.view(-1, p2d.shape[-2], p2d.shape[-1])
    mask_flat = mask.view(-1, mask.shape[-1])

    # Set masked out values to a large positive and negative value respectively
    # 将掩码外的值设置为一个大正数和一个大负数
    p2d_min = torch.where(mask_flat.unsqueeze(-1), p2d_flat, torch.tensor(float("inf")).to(p2d_flat))
    p2d_max = torch.where(mask_flat.unsqueeze(-1), p2d_flat, torch.tensor(float("-inf")).to(p2d_flat))

    # Compute the minimum and maximum coordinates for the bounding box
    # 计算边界框的最小和最大坐标
    xy_min = p2d_min.min(dim=1).values  # (BL, 2)
    xy_max = p2d_max.max(dim=1).values  # (BL, 2)

    # Concatenate min and max coordinates to form the bounding box
    # 将最小和最大坐标连接起来形成边界框
    bbx_xyxy = torch.cat([xy_min, xy_max], dim=-1)  # (BL, 4)

    # Reshape back to the original shape prefix
    # 将结果重新整形为原始形状前缀
    bbx_xyxy = bbx_xyxy.view(*p2d.shape[:-2], 4)

    return bbx_xyxy


def bbx_xyxy_ratio(xyxy1, xyxy2):
    """Designed for fov/unbounded
    计算两个边界框的面积比
    Args:
        xyxy1: (*, 4) 第一个边界框
        xyxy2: (*, 4) 第二个边界框
    Return:
        ratio: (*), squared_area(xyxy1) / squared_area(xyxy2) 面积比
    """
    area1 = (xyxy1[..., 2] - xyxy1[..., 0]) * (xyxy1[..., 3] - xyxy1[..., 1])
    area2 = (xyxy2[..., 2] - xyxy2[..., 0]) * (xyxy2[..., 3] - xyxy2[..., 1])
    # Check
    area1[~torch.isfinite(area1)] = 0  # replace inf in area1 with 0 将area1中的inf替换为0
    assert (area2 > 0).all(), "area2 should be positive"  # 确保area2为正
    return area1 / area2  # 返回面积比


def get_mesh_in_fov_category(mask):
    """mask: (L, V)
    获取网格在视场中的类别
    参数:
        mask: (L, V) 可见性掩码
    返回:
        class_type: 类别类型
        mask_frame_any_verts: 每帧是否有可见顶点的掩码
    The definition:
    1. FullyVisible: The mesh in every frame is entirely within the field of view (FOV). 网格在每一帧中都完全在视场内
    2. PartiallyVisible: In some frames, parts of the mesh are outside the FOV, while other parts are within the FOV. 在某些帧中,网格部分在视场外,部分在视场内。
    3. PartiallyOut: In some frames, the mesh is completely outside the FOV, while in others, it is visible. 在某些帧中,网格完全在视场外,而在其他帧中,它是可见的。
    4. FullyOut: The mesh is completely outside the FOV in every frame. 网格在每一帧中都完全在视场外。
    """
    mask = mask.clone().cpu()
    is_class1 = mask.all()  # FullyVisible 网格在每一帧中都完全在视场内
    is_class2 = mask.any(1).all() * ~is_class1  # PartiallyVisible 在某些帧中,网格部分在视场外,部分在视场内
    is_class4 = ~(mask.any())  # PartiallyOut 网格在每一帧中都完全在视场外
    is_class3 = ~is_class1 * ~is_class2 * ~is_class4  # FullyOut 

    mask_frame_any_verts = mask.any(1)
    assert is_class1.int() + is_class2.int() + is_class3.int() + is_class4.int() == 1
    class_type = is_class1.int() + 2 * is_class2.int() + 3 * is_class3.int() + 4 * is_class4.int()
    return class_type.item(), mask_frame_any_verts


def get_infov_mask(p2d, w_real, h_real):
    """
    获取在视场内的掩码
    Args:
        p2d: (B, L, V, 2) 2D点坐标
        w_real, h_real: (B, L) or int 真实图像宽度
    Returns:
        mask: (B, L, V) 在视场内的布尔掩码
    """
    x, y = p2d[..., 0], p2d[..., 1]
    if isinstance(w_real, int):
        mask = (x >= 0) * (x < w_real) * (y >= 0) * (y < h_real)
    else:
        mask = (x >= 0) * (x < w_real[..., None]) * (y >= 0) * (y < h_real[..., None])
    return mask

