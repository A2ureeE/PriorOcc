"""Cam4DOcc 评测适配器：网格上采样 + 类映射 + GMO/GSO 聚合。

物理范围对齐的事实（实测核实）：
  我们的 occ 网格:  200 x 200 x 16，pc_range = [-51.2,-51.2,-5, 51.2,51.2,3]
                    → 体素 0.512 x 0.512 x 0.5 m
  Cam4DOcc 网格:    512 x 512 x 40，pc_range 完全相同
                    → 体素 0.2 x 0.2 x 0.2 m
  因此上采样是纯索引最近邻，无需坐标重投影。

用法：
    from grid_adapter import upsample_occ, remap_to_nuscocc, gmo_mask, gso_mask
    pred512 = upsample_occ(pred200.int())          # (512,512,40)
    pred_occ = remap_to_nuscocc(pred512)           # 我们的 17(free) -> 0(empty)
"""
import numpy as np

# 我们的 18 类布局（CLASS_NAMES + free）
OUR_NUM_CLASSES = 18
OUR_FREE = 17
OUR_GMO = list(range(1, 11))    # barrier..truck（10 类，与 Cam4DOcc GMO 对齐）
OUR_GSO = list(range(11, 17))   # driveable_surface..vegetation（6 类）

# nuScenes-Occupancy（OpenOccupancy）：0=empty/noise, 1..16 语义（与我们的 1..16 恒等）
NUSCOCC_NUM_CLASSES = 17


def upsample_occ(pred, target=(512, 512, 40)):
    """最近邻上采样：200x200x16 -> 512x512x40（同物理范围）。

    src_idx = floor((dst_idx + 0.5) * src_size / dst_size)
    """
    pred = np.asarray(pred)
    src = pred.shape
    assert len(src) == 3, f'expect 3D occ grid, got {src}'
    axes = []
    for s, t in zip(src, target):
        idx = ((np.arange(t) + 0.5) * s / t).astype(np.int64)
        axes.append(np.clip(idx, 0, s - 1))
    return pred[np.ix_(axes[0], axes[1], axes[2])]


def downsample_occ(gt, target=(200, 200, 16), num_classes=OUR_NUM_CLASSES):
    """最近邻下采样（用于把 512^3 GT 映射到我们的网格做交叉验证）。

    仅用于验证映射正确性；正式评测应在上采样后的分辨率和官方 GT 上做。
    """
    gt = np.asarray(gt)
    src = gt.shape
    axes = []
    for s, t in zip(src, target):
        idx = ((np.arange(t) + 0.5) * s / t).astype(np.int64)
        axes.append(np.clip(idx, 0, s - 1))
    return gt[np.ix_(axes[0], axes[1], axes[2])]


def remap_to_nuscocc(pred):
    """我们的类 ID -> nuScenes-Occupancy 类 ID（1..16 恒等，free 17 -> 0）。"""
    out = np.asarray(pred).copy()
    out[out == OUR_FREE] = 0
    return out


def gmo_mask(pred):
    """布尔掩码：属于 GMO（可移动物体）的体素。"""
    p = np.asarray(pred)
    m = np.zeros_like(p, dtype=bool)
    for c in OUR_GMO:
        m |= (p == c)
    return m


def gso_mask(pred):
    p = np.asarray(pred)
    m = np.zeros_like(p, dtype=bool)
    for c in OUR_GSO:
        m |= (p == c)
    return m


def collapse_gmo(pred):
    """把 GMO 的 10 类合并为 1 类（Cam4DOcc 的 inflated GMO 口径）。"""
    p = np.asarray(pred).copy()
    m = gmo_mask(p)
    p[m] = 1
    return p


def _self_test():
    rng = np.random.RandomState(0)
    # 构造一个带结构的 200^3 预测：中心放一块 car(4)、底面放 driveable(11)
    pred = np.full((200, 200, 16), OUR_FREE, dtype=np.int64)
    pred[90:110, 90:110, 8:12] = 4        # car 块
    pred[:, :, 0:2] = 11                  # 地面
    up = upsample_occ(pred)
    print('[test] shape:', up.shape)
    assert up.shape == (512, 512, 40)
    # 上采样保体积：car 块体积比例应保持
    v_src = (pred == 4).sum() / pred.size
    v_dst = (up == 4).sum() / up.size
    print(f'[test] car 体积占比 src={v_src:.5f} dst={v_dst:.5f} 差={abs(v_src-v_dst):.5f}')
    assert abs(v_src - v_dst) < 2e-3, '体积占比不应显著漂移'
    # 类别集合不被破坏
    assert set(np.unique(up).tolist()) <= set(np.unique(pred).tolist())
    # free -> empty 重映射
    re = remap_to_nuscocc(up)
    assert set(np.unique(re).tolist()) <= set(range(NUSCOCC_NUM_CLASSES))
    print('[test] remap ok, unique =', sorted(np.unique(re).tolist()))
    # GMO/GSO 分组
    g = collapse_gmo(up)
    # GMO 体素并成 1；GSO 保留 11..16；free 仍为 17
    assert set(np.unique(g).tolist()) <= {1, 11, 12, 13, 14, 15, 16, 17}
    assert (g == 1).sum() == ((up >= 1) & (up <= 10)).sum(), 'GMO 合并后体素数应守恒'
    print('[test] GMO collapse ok, unique =', sorted(np.unique(g).tolist()))
    # 往返：上采样再下采样，结构应大致保持
    back = downsample_occ(up)
    # 结构保持检查：car 块中心仍在
    assert back[100, 100, 9] == 4 or back[100, 100, 10] == 4
    print('[test] roundtrip structure ok')
    print('[test] ALL PASS')


if __name__ == '__main__':
    _self_test()
