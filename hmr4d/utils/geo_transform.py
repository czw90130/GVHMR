import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pytorch3d.transforms import so3_exp_map, so3_log_map
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle, matrix_to_rotation_6d
import pytorch3d.ops.knn as knn
from hmr4d.utils.pylogger import Log
from pytorch3d.transforms import euler_angles_to_matrix
import hmr4d.utils.matrix as matrix
from einops import einsum, rearrange, repeat
from hmr4d.utils.geo.quaternion import qbetween

def homo_points(points):
    """
    将点转换为齐次坐标
    Args:
        points: (..., C) 输入点坐标
    Returns:
        (..., C+1) 齐次坐标,最后一维填充1
    """
    return F.pad(points, [0, 1], value=1.0)

def apply_Ts_on_seq_points(points, Ts):
    """
    perform translation matrix on related point
    对序列点应用变换矩阵
    Args:
        points: (..., N, 3) 输入点序列
        Ts: (..., N, 4, 4) 变换矩阵序列
    Returns: 
        (..., N, 3) 变换后的点序列
    """
    # 应用旋转和平移
    points = torch.torch.einsum("...ki,...i->...k", Ts[..., :3, :3], points) + Ts[..., :3, 3]
    return points

def apply_T_on_points(points, T):
    """
    对点应用单个变换矩阵
    Args:
        points: (..., N, 3) 输入点
        T: (..., 4, 4) 变换矩阵
    Returns:
        (..., N, 3) 变换后的点
    """
    # 应用旋转和平移
    points_T = torch.einsum("...ki,...ji->...jk", T[..., :3, :3], points) + T[..., None, :3, 3]
    return points_T

def T_transforms_points(T, points, pattern):
    """manual mode of apply_T_on_points
    手动模式的apply_T_on_points
    参数:
        T: (..., 4, 4) 变换矩阵
        points: (..., 3) 输入点
        pattern: "... c d, ... d -> ... c" einsum模式
    返回:
        变换后的点
    """
    return einsum(T, homo_points(points), pattern)[..., :3]

def project_p2d(points, K=None, is_pinhole=True):
    """
    将3D点投影到2D平面
    Args:
        points: (..., (N), 3) 3D点
        K: (..., 3, 3) 相机内参矩阵
        is_pinhole: 是否为针孔相机模型
    Returns: 
        shape is similar to points but without z 形状与输入points相似但没有z维度的2D点
    """
    points = points.clone()
    if is_pinhole:
        # 针孔相机模型
        z = points[..., [-1]]
        z.masked_fill_(z.abs() < 1e-6, 1e-6)  # 避免除以零
        points_proj = points / z
    else:  # orthogonal 正交投影
        points_proj = F.pad(points[..., :2], (0, 1), value=1)

    if K is not None:
        # Handle N
        # 应用相机内参
        if len(points_proj.shape) == len(K.shape):
            p2d_h = torch.einsum("...ki,...ji->...jk", K, points_proj)
        else:
            p2d_h = torch.einsum("...ki,...i->...k", K, points_proj)
    else:
        p2d_h = points_proj[..., :2]

    return p2d_h[..., :2]

def gen_uv_from_HW(H, W, device="cpu"):
    """
    生成图像坐标网格
    参数:
        H: 图像高度
        W: 图像宽度
        device: 计算设备
    返回: 
        (H, W, 2), as float. Note: uv not ij 浮点型坐标网格。注意：返回uv而非ij
    """
    grid_v, grid_u = torch.meshgrid(torch.arange(H), torch.arange(W))
    return (
        torch.stack(
            [grid_u, grid_v],
            dim=-1,
        )
        .float()
        .to(device)
    )  # (H, W, 2)

def unproject_p2d(uv, z, K):
    """we assume a pinhole camera for unprojection
    将2D点反投影到3D空间（假设针孔相机模型）
    参数:
        uv: (B, N, 2) 2D点坐标
        z: (B, N, 1) 深度值
        K: (B, 3, 3) 相机内参矩阵
    返回: 
        (B, N, 3) 3D点坐标
    """
    xy_atz1 = (uv - K[:, None, :2, 2]) / K[:, None, [0, 1], [0, 1]]  # (B, N, 2)
    xyz = torch.cat([xy_atz1 * z, z], dim=-1)
    return xyz

def cvt_p2d_from_i_to_c(uv, K):
    """
    将图像坐标系的2D点转换为相机坐标系
    Args:
        uv: (..., 2) 或 (..., N, 2) 图像坐标系的2D点
        K: (..., 3, 3) 相机内参矩阵
    Returns: 
        the same shape as input uv 与输入uv形状相同的相机坐标系2D点
    """
    if len(uv.shape) == len(K.shape):
        xy = (uv - K[..., None, :2, 2]) / K[..., None, [0, 1], [0, 1]]
    else:  # without N 没有N维度
        xy = (uv - K[..., :2, 2]) / K[..., [0, 1], [0, 1]]
    return xy

def cvt_to_bi01_p2d(p2d, bbx_lurb):
    """
    将2D点转换为边界框内的归一化坐标 (0-1范围)
    参数:
        p2d: (..., (N), 2) 2D点坐标
        bbx_lurb: (..., 4) 边界框坐标 (left, up, right, bottom)
    返回:
        归一化后的2D点坐标
    """
    if len(p2d.shape) == len(bbx_lurb.shape) + 1:
        bbx_lurb = bbx_lurb[..., None, :]

    bbx_wh = bbx_lurb[..., 2:] - bbx_lurb[..., :2]
    bi01_p2d = (p2d - bbx_lurb[..., :2]) / bbx_wh
    return bi01_p2d

def cvt_from_bi01_p2d(bi01_p2d, bbx_lurb):
    """Use bbx_lurb to resize bi01_p2d to p2d (image-coordinates)
    将边界框内的归一化坐标 (0-1范围) 转换回图像坐标系
    Args:
        bi01_p2d: (..., 2) 或 (..., N, 2) 归一化坐标
        bbx_lurb: (..., 4) 边界框坐标 (left, up, right, bottom)
    Returns:
        p2d: shape is the same as input
        与输入形状相同的图像坐标系2D点
    """
    bbx_wh = bbx_lurb[..., 2:] - bbx_lurb[..., :2]  # (..., 2)
    if len(bi01_p2d.shape) == len(bbx_wh.shape) + 1:
        p2d = (bi01_p2d * bbx_wh.unsqueeze(-2)) + bbx_lurb[..., None, :2]
    else:
        p2d = (bi01_p2d * bbx_wh) + bbx_lurb[..., :2]
    return p2d

def cvt_p2d_from_bi01_to_c(bi01, bbxs_lurb, Ks):
    """
    将边界框内的归一化坐标转换为相机坐标系
    Args:
        bi01: (..., (N), 2) value in range (0,1), the point in the bbx image 边界框内的归一化坐标 (0-1范围)
        bbxs_lurb: (..., 4) 边界框坐标
        Ks: (..., 3, 3) 相机内参矩阵
    Returns:
        c: (..., (N), 2) 相机坐标系下的2D点
    """
    i = cvt_from_bi01_p2d(bi01, bbxs_lurb)
    c = cvt_p2d_from_i_to_c(i, Ks)
    return c

def cvt_p2d_from_pm1_to_i(p2d_pm1, bbx_xys):
    """
    将[-1,1]范围内的归一化坐标转换为图像坐标系
    Args:
        p2d_pm1: (..., (N), 2) [-1,1]范围内的归一化坐标
        bbx_xys: (..., 3) 边界框信息 (中心x, 中心y, 大小)
    Returns:
        p2d: (..., (N), 2) 图像坐标系下的2D点
    """
    return bbx_xys[..., :2] + p2d_pm1 * bbx_xys[..., [2]] / 2

def uv2l_index(uv, W):
    """
    将uv坐标转换为线性索引
    参数:
        uv: (..., 2) uv坐标
        W: 图像宽度
    返回:
        线性索引
    """
    return uv[..., 0] + uv[..., 1] * W

def l2uv_index(l, W):
    """
    将线性索引转换为uv坐标
    参数:
        l: 线性索引
        W: 图像宽度
    返回:
        (..., 2) uv坐标
    """
    v = torch.div(l, W, rounding_mode="floor")
    u = l % W
    return torch.stack([u, v], dim=-1)

def transform_mat(R, t):
    """
    构建变换矩阵
    Args:
        R: Bx3x3 array of a batch of rotation matrices 旋转矩阵批次
        t: Bx3x(1) array of a batch of translation vectors 平移向量批次
    Returns:
        T: Bx4x4 Transformation matrix 变换矩阵
    """
    # No padding left or right, only add an extra row
    # 没有左右填充，只添加一个额外的行
    if len(R.shape) > len(t.shape):
        t = t[..., None]
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)

def axis_angle_to_matrix_exp_map(aa):
    """use pytorch3d so3_exp_map
    使用pytorch3d的so3_exp_map将轴角表示转换为旋转矩阵
    Args:
        aa: (*, 3) 轴角表示
    Returns:
        R: (*, 3, 3) 旋转矩阵
    """
    print("Use pytorch3d.transforms.axis_angle_to_matrix instead!!!") # 使用pytorch3d.transforms.axis_angle_to_matrix代替
    ori_shape = aa.shape[:-1]
    return so3_exp_map(aa.reshape(-1, 3)).reshape(*ori_shape, 3, 3)

def matrix_to_axis_angle_log_map(R):
    """use pytorch3d so3_log_map
    使用pytorch3d的so3_log_map将旋转矩阵转换为轴角表示
    Args:
        R: (*, 3, 3) 旋转矩阵
    Returns:
        aa: (*, 3) 轴角表示
    """
    print("WARINING! I met singularity problem with this function, use matrix_to_axis_angle instead!") # 警告！使用此函数时会遇到奇异性问题，请使用matrix_to_axis_angle代替！
    ori_shape = R.shape[:-2]
    return so3_log_map(R.reshape(-1, 3, 3)).reshape(*ori_shape, 3)

def matrix_to_axis_angle(R):
    """ use pytorch3d so3_log_map
    将旋转矩阵转换为轴角表示
    Args:
        R: (*, 3, 3) 旋转矩阵
    Returns:
        aa: (*, 3) 轴角表示
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(R))

def ransac_PnP(K, pts_2d, pts_3d, err_thr=10):
    """solve pnp
    使用RANSAC算法解决PnP问题
    参数:
        K: 相机内参矩阵
        pts_2d: 2D点坐标
        pts_3d: 3D点坐标
        err_thr: 重投影误差阈值
    返回:
        pose: 相机姿态 (3x4矩阵)
        pose_homo: 齐次形式的相机姿态 (4x4矩阵)
        inliers: 内点索引
    """
    dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

    pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
    pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64))
    K = K.astype(np.float64)

    try:
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, dist_coeffs, reprojectionError=err_thr, iterationsCount=10000, flags=cv2.SOLVEPNP_EPNP
        )

        rotation = cv2.Rodrigues(rvec)[0]

        pose = np.concatenate([rotation, tvec], axis=-1)
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

        inliers = [] if inliers is None else inliers

        return pose, pose_homo, inliers
    except cv2.error:
        print("CV ERROR")
        return np.eye(4)[:3], np.eye(4), []

def ransac_PnP_batch(K_raw, pts_2d, pts_3d, err_thr=10):
    """
    批量处理RANSAC PnP
    参数:
        K_raw: 相机内参矩阵批次
        pts_2d: 2D点坐标批次
        pts_3d: 3D点坐标批次
        err_thr: 重投影误差阈值
    返回:
        fit_R: 旋转矩阵批次
        fit_t: 平移向量批次
    """
    fit_R, fit_t = [], []
    for b in range(K_raw.shape[0]):
        pose, _, inliers = ransac_PnP(K_raw[b], pts_2d[b], pts_3d[b], err_thr=err_thr)
        fit_R.append(pose[:3, :3])
        fit_t.append(pose[:3, 3])
    fit_R = np.stack(fit_R, axis=0)
    fit_t = np.stack(fit_t, axis=0)
    return fit_R, fit_t

def triangulate_point(Ts_w2c, c_p2d, **kwargs):
    """
    三角化点（透视投影）
    """
    from hmr4d.utils.geo.triangulation import triangulate_persp

    print("Deprecated, please import from hmr4d.utils.geo.triangulation")
    return triangulate_persp(Ts_w2c, c_p2d, **kwargs)

def triangulate_point_ortho(Ts_w2c, c_p2d, **kwargs):
    """
    三角化点（正交投影）
    """
    from hmr4d.utils.geo.triangulation import triangulate_ortho

    print("Deprecated, please import from hmr4d.utils.geo.triangulation")
    return triangulate_ortho(Ts_w2c, c_p2d, **kwargs)

def get_nearby_points(points, query_verts, padding=0.0, p=1):
    """
    获取查询顶点附近的点
    参数:
        points: (S, 3) 所有点的坐标
        query_verts: (V, 3) 查询顶点的坐标
        padding: 边界填充
        p: 距离度量（1: L1距离, 2: L2距离）
    返回:
        nearby_points: 附近点的坐标
    """
    if p == 1:
        max_xyz = query_verts.max(0)[0] + padding
        min_xyz = query_verts.min(0)[0] - padding
        idx = (((points - min_xyz) > 0).all(dim=-1) * ((points - max_xyz) < 0).all(dim=-1)).nonzero().squeeze(-1)
        nearby_points = points[idx]
    elif p == 2:
        squared_dist, _, _ = knn.knn_points(points[None], query_verts[None], K=1, return_nn=False)
        mask = squared_dist[0, :, 0] < padding**2  # (S,)
        nearby_points = points[mask]

    return nearby_points

def unproj_bbx_to_fst(bbx_lurb, K, near_z=0.5, far_z=12.5):
    """
    将2D边界框反投影到3D视锥体
    参数:
        bbx_lurb: (B, 4) 2D边界框坐标 (left, up, right, bottom)
        K: (B, 3, 3) 相机内参矩阵
        near_z: 近平面距离
        far_z: 远平面距离
    返回:
        c_frustum_points: (B, 8, 3) 3D视锥体的8个顶点
    """
    B = bbx_lurb.size(0)
    uv = bbx_lurb[:, [[0, 1], [2, 1], [2, 3], [0, 3], [0, 1], [2, 1], [2, 3], [0, 3]]]
    if isinstance(near_z, float):
        z = uv.new([near_z] * 4 + [far_z] * 4).reshape(1, 8, 1).repeat(B, 1, 1)
    else:
        z = torch.cat([near_z[:, None, None].repeat(1, 4, 1), far_z[:, None, None].repeat(1, 4, 1)], dim=1)
    c_frustum_points = unproject_p2d(uv, z, K)  # (B, 8, 3)
    return c_frustum_points

def convert_bbx_xys_to_lurb(bbx_xys):
    """
    将中心点和大小表示的边界框转换为左上右下表示
    bbx_xys (..., 3) -> bbx_lurb (..., 4)
    Args:
        bbx_xys: (..., 3) [center_x, center_y, size]
    Returns:
        lurb: (..., 4) [left, up, right, bottom]
    """
    size = bbx_xys[..., 2:]
    center = bbx_xys[..., :2]
    lurb = torch.cat([center - size / 2, center + size / 2], dim=-1)
    return lurb

def convert_lurb_to_bbx_xys(bbx_lurb):
    """bbx_lurb (..., 4) -> bbx_xys (..., 3) be aware that it is squared
    将左上右下表示的边界框转换为中心点和大小表示
    Args:
        bbx_lurb: (..., 4) [left, up, right, bottom]
    Returns:
        bbx_xys: (..., 3) [center_x, center_y, size] 注意size是正方形的
    """
    size = (bbx_lurb[..., 2:] - bbx_lurb[..., :2]).max(-1, keepdim=True)[0]
    center = (bbx_lurb[..., :2] + bbx_lurb[..., 2:]) / 2
    return torch.cat([center, size], dim=-1)


# ================== AZ/AY Transformations ================== #
# AZ/AY 坐标系变换

# This section contains functions for transforming between different coordinate systems,
# specifically AZ (Azimuth) and AY (Azimuth-Y) coordinate systems.
# 本节包含在不同坐标系之间进行变换的函数，特别是AZ（方位角）和AY（方位角-Y）坐标系。

# These transformations are crucial for aligning 3D human poses and camera views
# in various computer vision and graphics applications.
# 这些变换对于在各种计算机视觉和图形应用中对齐3D人体姿势和相机视图至关重要。

# The functions below implement conversions between AZ and AY coordinate systems,
# which are commonly used in human pose estimation and 3D reconstruction tasks.
# 以下函数实现了AZ和AY坐标系之间的转换，这些坐标系常用于人体姿势估计和3D重建任务。


def compute_T_ayf2az(joints, inverse=False):
    """
    计算从ayf坐标系到az坐标系的变换矩阵
    Args:
        joints: (B, J, 3), in the start-frame, az-coordinate 在起始帧中的关节坐标，使用az坐标系
    Returns:
        if inverse == False:
        else:
           T_af2az: (B, 4, 4)
        else :
            T_az2af: (B, 4, 4)
    """

    t_ayf2az = joints[:, 0, :].detach().clone() # 获取根关节位置作为平移向量
    t_ayf2az[:, 2] = 0  # do not modify z 不修改z轴坐标

    RL_xy_h = joints[:, 1, [0, 1]] - joints[:, 2, [0, 1]]  # (B, 2), hip point to left side 计算髋部左侧点的xy坐标差
    RL_xy_s = joints[:, 16, [0, 1]] - joints[:, 17, [0, 1]]  # (B, 2), shoulder point to left side 计算肩部左侧点的xy坐标差
    RL_xy = RL_xy_h + RL_xy_s  # 合并髋部和肩部的左侧方向
    I_mask = RL_xy.pow(2).sum(-1) < 1e-4  # do not rotate, when can't decided the face direction 当无法确定朝向时不旋转
    if I_mask.sum() > 0:
        Log.warn("{} samples can't decide the face direction".format(I_mask.sum()))  # 记录无法确定朝向的样本数量
    x_dir = F.pad(F.normalize(RL_xy, 2, -1), (0, 1), value=0)  # (B, 3) 计算并填充x方向向量
    y_dir = torch.zeros_like(x_dir)  # 初始化y方向向量
    y_dir[..., 2] = 1  # 设置y方向向量的z分量为1
    z_dir = torch.cross(x_dir, y_dir, dim=-1)  # 计算z方向向量（叉乘）
    R_ayf2az = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (B, 3, 3) 构建旋转矩阵
    R_ayf2az[I_mask] = torch.eye(3).to(R_ayf2az)  # 对无法确定朝向的样本使用单位矩阵

    if inverse:
        R_az2ayf = R_ayf2az.transpose(1, 2)  # (B, 3, 3) 计算逆旋转矩阵
        t_az2ayf = -einsum(R_ayf2az, t_ayf2az, "b i j , b i -> b j")  # (B, 3) 计算逆平移向量
        return transform_mat(R_az2ayf, t_az2ayf)  # 返回逆变换矩阵
    else:
        return transform_mat(R_ayf2az, t_ayf2az)  # 返回正向变换矩阵


def compute_T_ayfz2ay(joints, inverse=False):
    """
    计算从ayfz坐标系到ay坐标系的变换矩阵
    Args:
        joints: (B, J, 3), in the start-frame, ay-coordinate 起始帧中的关节坐标，使用ay坐标系
    Returns:
        if inverse == False:
            T_ayfz2ay: (B, 4, 4) 从ayfz到ay的变换矩阵
        else :
            T_ay2ayfz: (B, 4, 4) 从ay到ayfz的变换矩阵
    """
    t_ayfz2ay = joints[:, 0, :].detach().clone()  # 获取根关节位置作为平移向量
    t_ayfz2ay[:, 1] = 0  # do not modify y 不修改y轴坐标

    RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]  # (B, 2), hip point to left side 计算髋部左侧点的xz坐标差
    RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]  # (B, 2), shoulder point to left side 计算肩部左侧点的xz坐标差
    RL_xz = RL_xz_h + RL_xz_s  # 合并髋部和肩部的左侧方向
    I_mask = RL_xz.pow(2).sum(-1) < 1e-4  # do not rotate, when can't decided the face direction 当无法确定朝向时不旋转
    if I_mask.sum() > 0:
        Log.warn("{} samples can't decide the face direction".format(I_mask.sum()))  # 记录无法确定朝向的样本数量

    x_dir = torch.zeros_like(t_ayfz2ay)  # (B, 3) 初始化x方向向量
    x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)  # 计算并归一化x方向向量
    y_dir = torch.zeros_like(x_dir)  # 初始化y方向向量
    y_dir[..., 1] = 1  # (B, 3) 设置y方向向量的y分量为1
    z_dir = torch.cross(x_dir, y_dir, dim=-1)  # 计算z方向向量（叉乘）
    R_ayfz2ay = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (B, 3, 3) 构建旋转矩阵
    R_ayfz2ay[I_mask] = torch.eye(3).to(R_ayfz2ay)  # 对无法确定朝向的样本使用单位矩阵

    if inverse:
        R_ay2ayfz = R_ayfz2ay.transpose(1, 2)  # (B, 3, 3) 计算逆旋转矩阵
        t_ay2ayfz = -einsum(R_ayfz2ay, t_ayfz2ay, "b i j , b i -> b j")  # (B, 3) 计算逆平移向量
        return transform_mat(R_ay2ayfz, t_ay2ayfz)  # 返回逆变换矩阵
    else:
        return transform_mat(R_ayfz2ay, t_ayfz2ay)  # 返回正向变换矩阵


def compute_T_ay2ayrot(joints):
    """
    计算从ay坐标系到ayrot坐标系的变换矩阵
    Args:
        joints: (B, J, 3), in the start-frame, ay-coordinate，在起始帧中的关节坐标，使用ay坐标系
    Returns:
        T_ay2ayrot: (B, 4, 4) 从ay坐标系到ayrot坐标系的4x4变换矩阵
    """
    t_ayrot2ay = joints[:, 0, :].detach().clone()
    # 复制根关节位置作为平移向量
    t_ayrot2ay[:, 1] = 0  # do not modify y
    # 不修改y轴坐标

    B = joints.shape[0]
    euler_angle = torch.zeros((B, 3), device=joints.device)
    # 初始化欧拉角张量
    yrot_angle = torch.rand((B,), device=joints.device) * 2 * torch.pi
    # 生成随机的y轴旋转角度
    euler_angle[:, 0] = yrot_angle
    # 将y轴旋转角度赋值给欧拉角的第一个分量
    R_ay2ayrot = euler_angles_to_matrix(euler_angle, "YXZ")  # (B, 3, 3)
    # 将欧拉角转换为旋转矩阵

    R_ayrot2ay = R_ay2ayrot.transpose(1, 2)
    # 计算逆旋转矩阵
    t_ay2ayrot = -einsum(R_ayrot2ay, t_ayrot2ay, "b i j , b i -> b j")
    # 计算平移向量
    return transform_mat(R_ay2ayrot, t_ay2ayrot)
    # 返回变换矩阵


def compute_root_quaternion_ay(joints):
    """
    计算从ay坐标系到面朝前方向的根部四元数旋转
    Args:
        joints: (B, J, 3), in the start-frame, ay-coordinate 在起始帧中的关节坐标，使用ay坐标系
    Returns:
        root_quat: (B, 4) from z-axis to fz 从z轴到面朝前方向的四元数
    """
    joints_shape = joints.shape
    joints = joints.reshape((-1,) + joints_shape[-2:])  # 重塑关节张量以便批处理
    t_ayfz2ay = joints[:, 0, :].detach().clone()  # 提取根关节位置
    t_ayfz2ay[:, 1] = 0  # do not modify y 不修改y轴坐标，保持高度不变

    RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]  # (B, 2), hip point to left side # 计算髋部左右方向向量
    RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]  # (B, 2), shoulder point to left side # 计算肩部左右方向向量
    RL_xz = RL_xz_h + RL_xz_s  # 合并髋部和肩部向量以获得更稳定的左右方向
    I_mask = RL_xz.pow(2).sum(-1) < 1e-4  # do not rotate, when can't decided the face direction
                                          # 当无法确定朝向时不进行旋转
    if I_mask.sum() > 0:
        Log.warn("{} samples can't decide the face direction".format(I_mask.sum()))  # 记录无法确定朝向的样本数量

    x_dir = torch.zeros_like(t_ayfz2ay)  # (B, 3)  # 初始化x方向向量
    x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)  # 计算并归一化x方向向量
    y_dir = torch.zeros_like(x_dir)  # 初始化y方向向量
    y_dir[..., 1] = 1  # (B, 3)  # 设置y方向向量的y分量为1
    z_dir = torch.cross(x_dir, y_dir, dim=-1)  # 计算z方向向量（叉乘）

    z_dir[..., 2] += 1e-9  # 添加小偏移量以避免数值不稳定
    pos_z_vec = torch.tensor([0, 0, 1]).to(joints.device).float()  # (3,)  # 创建正z轴向量
    root_quat = qbetween(pos_z_vec[None], z_dir)  # (B, 4)  # 计算从正z轴到面朝前方向的四元数
    root_quat = root_quat.reshape(joints_shape[:-2] + (4,))  # 重塑四元数张量以匹配输入形状
    return root_quat  # 返回计算得到的根部四元数


# ================== Transformations between two sets of features ================== #
# 两组特征之间的变换

# This section contains functions for computing transformations between two sets of features.
# 本节包含用于计算两组特征之间变换的函数。

# These transformations are essential for aligning different coordinate systems or point sets,
# which is crucial in various computer vision and graphics applications.
# 这些变换对于对齐不同坐标系或点集至关重要，在各种计算机视觉和图形应用中都非常重要。

# The functions below implement methods such as Procrustes analysis and similarity transforms,
# which are commonly used to find the optimal rotation, translation, and scaling between two sets of corresponding points.
# 以下函数实现了诸如Procrustes分析和相似变换等方法，这些方法常用于找到两组对应点之间的最优旋转、平移和缩放。


def similarity_transform_batch(S1, S2):
    """
    Computes a similarity transform (sR, t) that solves the orthogonal Procrutes problem.
    计算解决正交Procrustes问题的相似变换(sR, t)。
    Args:
        S1, S2: (*, L, 3)
    """
    assert S1.shape == S2.shape
    S_shape = S1.shape
    S1 = S1.reshape(-1, *S_shape[-2:])  # 重塑S1为二维批处理形式
    S2 = S2.reshape(-1, *S_shape[-2:])  # 重塑S2为二维批处理形式

    S1 = S1.transpose(-2, -1)  # 转置S1，使点集成为列向量
    S2 = S2.transpose(-2, -1)  # 转置S2，使点集成为列向量

    # --- The code is borrowed from WHAM ---
    # 代码借鉴自WHAM项目

    # 1. Remove mean.
    # 1. 移除均值
    # This step centers the point sets by subtracting their respective means.
    # 这一步通过减去各自的均值来使点集居中
    # Centering is crucial for accurate rotation and scale estimation.
    # 居中操作对于准确估计旋转和缩放至关重要
    mu1 = S1.mean(axis=-1, keepdims=True)  # axis is along N, S1(B, 3, N)
    # 计算S1的均值，保持维度
    mu2 = S2.mean(axis=-1, keepdims=True)
    # 计算S2的均值，保持维度

    X1 = S1 - mu1  # 中心化S1
    X2 = S2 - mu2  # 中心化S2

    # 2. Compute variance of X1 used for scale.
    # 2. 计算X1的方差，用于后续的尺度计算
    # This step calculates the variance of X1, which will be used to determine the scaling factor.
    # 这一步计算X1的方差，将用于确定缩放因子
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    # 3. 计算X1和X2的外积
    # This step computes the covariance matrix between X1 and X2, which is crucial for finding the optimal rotation.
    # 这一步计算X1和X2之间的协方差矩阵，这对于找到最优旋转至关重要
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    # 4. 最大化trace(R'K)的解为R=U*V'，其中U和V是K的奇异向量
    # This step performs Singular Value Decomposition (SVD) on K to find the optimal rotation.
    # 这一步对K进行奇异值分解(SVD)以找到最优旋转
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    # 构造Z矩阵以确保R的行列式为1
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    # 构造旋转矩阵R
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    # 5. 恢复尺度因子
    # This step calculates the scaling factor using the trace of R*K and the variance of X1.
    # 这一步使用R*K的迹和X1的方差来计算缩放因子
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    # 6. 恢复平移向量
    # This step computes the translation vector using the means, scale, and rotation.
    # 这一步使用均值、尺度和旋转来计算平移向量
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # -------
    # reshape back
    # Reshape the scale, rotation, and translation matrices back to their original shapes
    # 将尺度、旋转和平移矩阵重塑回原始形状
    # sR = scale[:, None, None] * R
    # Multiply scale with rotation matrix (commented out for now)
    # 将尺度与旋转矩阵相乘（目前被注释掉）
    # sR = sR.reshape(*S_shape[:-2], 3, 3)
    # Reshape the scaled rotation matrix (commented out for now)
    # 重塑缩放后的旋转矩阵（目前被注释掉）
    scale = scale.reshape(*S_shape[:-2], 1, 1)  # 重塑尺度因子
    R = R.reshape(*S_shape[:-2], 3, 3)  # 重塑旋转矩阵
    t = t.reshape(*S_shape[:-2], 3, 1)  # 重塑平移向量

    return (scale, R), t  # 返回尺度因子、旋转矩阵和平移向量


def kabsch_algorithm_batch(X1, X2):
    """
    Computes a rigid transform (R, t)
    计算刚体变换 (R, t)
    Args:
        X1, X2: (*, L, 3)
    """
    assert X1.shape == X2.shape
    X_shape = X1.shape
    X1 = X1.reshape(-1, *X_shape[-2:])  # 重塑X1为二维矩阵，保留最后两个维度
    X2 = X2.reshape(-1, *X_shape[-2:])  # 重塑X2为二维矩阵，保留最后两个维度

    # 1. 计算质心
    centroid_X1 = torch.mean(X1, dim=-2, keepdim=True)  # 计算X1的质心
    centroid_X2 = torch.mean(X2, dim=-2, keepdim=True)  # 计算X2的质心

    # 2. 去中心化
    X1_centered = X1 - centroid_X1  # 将X1中心化
    X2_centered = X2 - centroid_X2  # 将X2中心化

    # 3. 计算协方差矩阵
    H = torch.matmul(X1_centered.transpose(-2, -1), X2_centered)  # 计算X1和X2的协方差矩阵

    # 4. 奇异值分解
    U, S, Vt = torch.linalg.svd(H)  # 对协方差矩阵进行奇异值分解

    # 5. 计算旋转矩阵
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))  # 计算初始旋转矩阵

    # 修正反射矩阵
    d = (torch.det(R) < 0).unsqueeze(-1).unsqueeze(-1)  # 检测是否需要修正反射矩阵
    Vt = torch.where(d, -Vt, Vt)  # 如果需要，修正Vt
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))  # 重新计算旋转矩阵

    # 6. 计算平移向量
    t = centroid_X2.transpose(-2, -1) - torch.matmul(R, centroid_X1.transpose(-2, -1))  # 计算平移向量

    # -------
    # reshape back
    R = R.reshape(*X_shape[:-2], 3, 3)  # 将旋转矩阵重塑回原始形状
    t = t.reshape(*X_shape[:-2], 3, 1)  # 将平移向量重塑回原始形状

    return R, t  # 返回旋转矩阵和平移向量


# ===== WHAM cam_angvel ===== #
# WHAM相机角速度计算

# This section contains functions for computing camera angular velocity
# 本节包含用于计算相机角速度的函数

# Camera angular velocity is crucial for estimating camera motion in video sequences
# 相机角速度对于估计视频序列中的相机运动至关重要

# The functions below implement methods to calculate angular velocity from camera rotation matrices
# 以下函数实现了从相机旋转矩阵计算角速度的方法


def compute_cam_angvel(R_w2c, padding_last=True):
    """
    计算相机的角速度

    参数:
    R_w2c : (F, 3, 3) 世界坐标系到相机坐标系的旋转矩阵序列
    padding_last : bool, 是否在最后添加填充

    返回:
    cam_angvel : (F, 6) 相机的角速度序列
    """
    # R @ R0 = R1, so R = R1 @ R0^T
    # R @ R0 = R1，所以 R = R1 @ R0^T
    cam_angvel = matrix_to_rotation_6d(R_w2c[1:] @ R_w2c[:-1].transpose(-1, -2))  # (F-1, 6)
    # 计算相邻帧之间的旋转，并转换为6D表示

    # cam_angvel = (cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]])) * FPS
    # 注释掉的代码：可能用于调整角速度的尺度或基准

    assert padding_last
    # 确保padding_last为True，以便在序列末尾添加填充

    cam_angvel = torch.cat([cam_angvel, cam_angvel[-1:]], dim=0)  # (F, 6)
    # 在序列末尾添加最后一帧的复制，使输出维度与输入一致

    return cam_angvel.float()
    # 返回浮点类型的角速度序列


def ransac_gravity_vec(xyz, num_iterations=100, threshold=0.05, verbose=False):
    """
    使用RANSAC算法估计重力向量
    Args:
        xyz: (L, 3) 输入的3D点云数据
        num_iterations: int, RANSAC迭代次数
        threshold: float, 内点判定阈值
        verbose: bool, 是否打印详细信息
    Returns:
        result: (3,) 估计的重力向量
        max_inliers: 最佳模型的内点
    """
    # xyz: (L, 3)
    N = xyz.shape[0]  # 获取输入数据的点数
    max_inliers = []  # 存储最大内点集
    best_model = None  # 存储最佳模型
    norms = xyz.norm(dim=-1)  # (L,) 计算每个点的范数

    for _ in range(num_iterations):
        # 随机选择一个样本
        sample_index = np.random.randint(N)  # 随机选择一个点的索引
        sample = xyz[sample_index]  # (3,) 获取选中的样本点

        # 计算所有点与样本点的角度差
        dot_product = (xyz * sample).sum(dim=-1)  # (L,) 计算点积
        angles = dot_product / norms * norms[sample_index]  # (L,) 计算角度余弦值
        angles = torch.clamp(angles, -1, 1)  # 防止数值误差导致的异常，将值限制在[-1, 1]范围内
        angles = torch.acos(angles)  # 计算角度

        # 确定内点
        inliers = xyz[angles < threshold]  # 根据角度阈值筛选内点

        if len(inliers) > len(max_inliers):  # 如果当前内点数量大于最大内点数量
            max_inliers = inliers  # 更新最大内点集
            best_model = sample  # 更新最佳模型
        if len(max_inliers) == N:  # 如果所有点都是内点，提前结束迭代
            break
    
    if verbose:
        print(f"Inliers: {len(max_inliers)} / {N}")  # 打印内点数量和总点数
    
    result = max_inliers.mean(dim=0)  # 计算最大内点集的平均值作为最终结果

    return result, max_inliers  # 返回估计的重力向量和最大内点集

# NOTE: 最佳相机估计
# 用于获取序列中最佳的相机估计
def sequence_best_cammat(w_j3d, c_j3d, cam_rot):
    """
    获取序列中最佳的相机估计，要求静态相机
    Get the best camera estimation along the sequence, requires static camera
    Args:
        w_j3d: (L, J, 3) 世界坐标系下的3D关节点坐标
        c_j3d: (L, J, 3) 相机坐标系下的3D关节点坐标
        cam_rot: (L, 3, 3) 相机旋转矩阵
    Returns:
        cam_mat: (4, 4) 最佳相机矩阵
        ind: int 最佳相机矩阵对应的帧索引
    """
    L, J, _ = w_j3d.shape  # 获取序列长度L和关节点数量J

    root_in_w = w_j3d[:, 0]  # (L, 3) 提取世界坐标系下的根关节点坐标
    root_in_c = c_j3d[:, 0]  # (L, 3) 提取相机坐标系下的根关节点坐标
    cam_mat = matrix.get_TRS(cam_rot, root_in_w)  # (L, 4, 4) 计算相机矩阵
    cam_pos = matrix.get_position_from(-root_in_c[:, None], cam_mat)[:, 0]  # (L, 3) 计算相机位置
    cam_mat = matrix.set_position(cam_mat, cam_pos)  # (L, 4, 4) 更新相机矩阵，设置正确的相机位置

    w_j3d_expand = w_j3d[None].expand(L, -1, -1, -1)  # (L, L, J, 3) 扩展世界坐标系下的3D关节点坐标
    w_j3d_expand = w_j3d_expand.reshape(L, -1, 3)  # (L, L*J, 3) 重塑张量以便于后续计算

    # 计算重投影误差
    w_j3d_expand_in_c = matrix.get_relative_position_to(w_j3d_expand, cam_mat)  # (L, L*J, 3) 将世界坐标转换到相机坐标系
    w_j2d_expand_in_c = project_p2d(w_j3d_expand_in_c)  # (L, L*J, 2) 将3D点投影到2D平面
    w_j2d_expand_in_c = w_j2d_expand_in_c.reshape(L, L, J, 2)  # (L, L, J, 2) 重塑张量以匹配原始形状
    c_j2d = project_p2d(c_j3d)  # (L, J, 2) 将相机坐标系下的3D点投影到2D平面
    error = w_j2d_expand_in_c - c_j2d[None]  # (L, L, J, 2) 计算投影误差
    error = error.norm(dim=-1).mean(dim=-1)  # (L, L) 计算每帧的平均误差
    error = error.mean(dim=-1)  # (L,) 计算每个相机估计的总体误差
    ind = error.argmin()  # 找到误差最小的帧索引
    return cam_mat[ind], ind  # 返回最佳相机矩阵和对应的帧索引


# NOTE: 估计外参
# 用于获取序列的相机外参矩阵。
def get_sequence_cammat(w_j3d, c_j3d, cam_rot):
    """
    获取序列的相机外参矩阵
    Args:
        w_j3d: (L, J, 3) 世界坐标系下的3D关节点坐标
        c_j3d: (L, J, 3) 相机坐标系下的3D关节点坐标
        cam_rot: (L, 3, 3) 相机旋转矩阵
    Returns:
        cam_mat: (L, 4, 4) 相机外参矩阵
    """
    # w_j3d: (L, J, 3)
    # c_j3d: (L, J, 3)
    # cam_rot: (L, 3, 3)

    L, J, _ = w_j3d.shape
    # 获取序列长度L和关节点数量J

    root_in_w = w_j3d[:, 0]  # (L, 3)
    # 提取世界坐标系下的根关节点坐标
    root_in_c = c_j3d[:, 0]  # (L, 3)
    # 提取相机坐标系下的根关节点坐标
    cam_mat = matrix.get_TRS(cam_rot, root_in_w)  # (L, 4, 4)
    # 使用相机旋转矩阵和世界坐标系下的根关节点坐标计算初始相机矩阵
    cam_pos = matrix.get_position_from(-root_in_c[:, None], cam_mat)[:, 0]  # (L, 3)
    # 计算相机在世界坐标系中的位置
    cam_mat = matrix.set_position(cam_mat, cam_pos)  # (L, 4, 4)
    # 更新相机矩阵，设置正确的相机位置
    return cam_mat
    # 返回计算得到的相机外参矩阵


def ransac_vec(vel, min_multiply=20, verbose=False):
    """
    使用RANSAC算法去除速度向量中的异常值
    Args:
        vel: (L, 3) 速度向量序列
        min_multiply: 阈值倍数
        verbose: 是否打印详细信息
    Returns:
        result: (3,) 平均速度向量
        inner_mask: (L,) 内点掩码
    """
    # xyz: (L, 3)
    # remove outlier velocity
    # 移除异常速度值
    N = vel.shape[0]  # 获取速度向量序列的长度
    vel_1 = vel[None].expand(N, -1, -1)  # (L, L, 3) 扩展vel为(L, L, 3)形状
    vel_2 = vel[:, None].expand(-1, N, -1)  # (L, L, 3) 扩展vel为(L, L, 3)形状
    dist_mat = (vel_1 - vel_2).norm(dim=-1)  # (L, L) 计算每对速度向量之间的距离
    big_identity = torch.eye(N, device=vel.device) * 1e6  # 创建一个大的对角矩阵
    dist_mat_ = dist_mat + big_identity  # 将对角线元素设置为很大的值，避免自身比较
    threshold = dist_mat_.min() * min_multiply  # 计算阈值
    inner_mask = dist_mat < threshold  # (L, L) 创建内点掩码
    inner_num = inner_mask.sum(dim=-1)  # (L, ) 计算每个向量的内点数量
    ind = inner_num.argmax()  # 找到内点数量最多的向量索引
    result = vel[inner_mask[ind]].mean(dim=0)  # (3,) 计算内点的平均速度向量
    if verbose:
        print(inner_mask[ind].sum().item())  # 如果verbose为True，打印内点数量

    return result, inner_mask[ind]  # 返回平均速度向量和内点掩码
