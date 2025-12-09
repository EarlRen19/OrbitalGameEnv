import ctypes
import numpy as np
import os

# 1. 加载编译好的 .so 文件
# 确保路径相对于此文件或使用绝对路径
try:
    # 使用绝对路径加载
    so_path = "/home/star/Downloads/gemini-cli-main/OrbitalGameEnv/demos/OrbitLib/so/X86/libOrbit.so"
    if not os.path.exists(so_path):
        raise FileNotFoundError(f"Shared library not found at: {so_path}")
    orbit_lib_c = ctypes.CDLL(so_path)
except (FileNotFoundError, OSError) as e:
    print(f"\033[91mError loading libOrbit.so: {e}\033[0m")
    print("\033[93mWarning: LVLH transformations will be disabled. Falling back to ECI coordinates.\033[0m")
    orbit_lib_c = None

if orbit_lib_c:
    # 2. 定义 C 函数的参数类型
    # void DCM_J2000_to_LVLH(const double RV[6], double DCM[3][3]);

    # 定义 double 数组类型
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_double_3x3 = (ctypes.c_double * 3) * 3  # 二维数组

    orbit_lib_c.DCM_J2000_to_LVLH.argtypes = [
        c_double_p,  # RV[6]
        ctypes.POINTER(c_double_3x3)      # DCM[3][3] (输出)
    ]
    orbit_lib_c.DCM_J2000_to_LVLH.restype = None

def get_lvlh_dcm(state_j2000: np.ndarray) -> np.ndarray | None:
    """
    计算从 J2000 到 LVLH 的旋转矩阵 (DCM)
    Args:
        state_j2000: np.array, shape=(6,), [x, y, z, vx, vy, vz] (ECI Frame)
    Returns:
        dcm: np.array, shape=(3, 3) or None if library is not loaded
    """
    if not orbit_lib_c:
        return None

    # 准备输入数据
    rv_in = state_j2000.astype(np.float64)
    rv_in_p = rv_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    dcm_out = c_double_3x3()
    
    # 调用 C 函数
    orbit_lib_c.DCM_J2000_to_LVLH(rv_in_p, dcm_out)
    
    # 将结果转换为 numpy 数组
    dcm_np = np.array([[dcm_out[i][j] for j in range(3)] for i in range(3)])
    
    return dcm_np

def eci_to_lvlh_relative(observer_state: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """
    计算目标相对于观察者的 LVLH 坐标
    Args:
        observer_state: 追击者状态 (6,) ECI
        target_pos: 目标位置 (3,) ECI
        target_vel: 目标速度 (3,) ECI (可选)
    Returns:
        rel_pos_lvlh: (3,)
        rel_vel_lvlh: (3,) or None
    """
    # 1. 获取旋转矩阵 (DCM)
    # 这个矩阵可以将 ECI 坐标系下的向量 旋转到 LVLH 坐标系
    dcm = get_lvlh_dcm(observer_state)
    
    # 如果库加载失败，返回原始ECI相对坐标以保证程序运行
    if dcm is None:
        observer_pos = observer_state[:3]
        rel_pos_eci = target_pos - observer_pos
        rel_vel_eci = None
        if target_vel is not None:
            observer_vel = observer_state[3:]
            rel_vel_eci = target_vel - observer_vel
        return rel_pos_eci, rel_vel_eci

    # 2. 计算 ECI 系下的相对向量
    observer_pos = observer_state[:3]
    rel_pos_eci = target_pos - observer_pos
    
    # 3. 旋转位置向量: R_lvlh = DCM * R_eci
    rel_pos_lvlh = dcm @ rel_pos_eci
    
    rel_vel_lvlh = None
    if target_vel is not None:
        observer_vel = observer_state[3:]
        rel_vel_eci = target_vel - observer_vel
        
        # 注意：严格来说相对速度转换包含科里奥利项 (omega x r)，
        # 但在强化学习观测中，通常直接旋转相对速度矢量就足够让网络学习了。
        # 这里我们直接做投影：
        rel_vel_lvlh = dcm @ rel_vel_eci
        
    return rel_pos_lvlh, rel_vel_lvlh
