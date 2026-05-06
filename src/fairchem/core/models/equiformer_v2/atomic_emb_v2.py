"""
AtomPropertyEncoderV2 — 重设计版原子特征编码器
物理驱动特征集，针对 MOF CO₂/H₂O 吸附预测优化。
接口与 AtomPropertyEncoder（V1）完全一致：
  forward(atomic_numbers[N], tags[N]) -> [N, 128]
"""
import torch
import torch.nn as nn

# 价层 d 电子数查表（中性原子基态）
VALENCE_D_ELECTRONS = {
    # 第四周期过渡金属
    22: 2,   # Ti
    23: 3,   # V
    24: 5,   # Cr（d⁵ 半满）
    25: 5,   # Mn²⁺ hs
    26: 6,   # Fe
    27: 7,   # Co
    28: 8,   # Ni
    29: 10,  # Cu（d¹⁰）
    30: 10,  # Zn
    # 第五周期
    40: 2,   # Zr
    42: 6,   # Mo
    # 主族金属（无 d 电子）
    13: 0,   # Al
    49: 0,   # In
    # 有机连接子常见非金属
    1:  0,   # H
    6:  0,   # C
    7:  0,   # N
    8:  0,   # O
}

# 常见 OMS 金属（开放金属位点）
OMS_METALS = {29, 30, 26, 24, 25, 28, 27, 42}  # Cu, Zn, Fe, Cr, Mn, Ni, Co, Mo

# p 区金属
P_BLOCK_METALS = {13, 31, 49, 50, 81, 82, 83, 84}


def _safe_get(x, default=0.0):
    try:
        if x is None:
            return default
        if callable(x):
            x = x()
        return float(x) if x is not None else default
    except Exception:
        return default


def _get_ie(el, n=1):
    """获取第 n 电离能（eV），使用 _ionization_energies 规避版本 bug"""
    try:
        ies = el._ionization_energies
        if not ies:
            return 0.0
        # 按 degree 查找
        for ie in ies:
            if getattr(ie, 'degree', None) == n:
                return _safe_get(getattr(ie, 'energy', None))
        # Fallback：排序后取第 n 个
        sorted_ies = sorted(ies, key=lambda x: getattr(x, 'degree', 0))
        if len(sorted_ies) >= n:
            return _safe_get(getattr(sorted_ies[n - 1], 'energy', None))
        return 0.0
    except Exception:
        return 0.0


def _get_ionic_radius_VI(el):
    """获取六配位（八面体）离子半径，单位 pm（mendeleev 存 Å，乘100转换）"""
    try:
        ir_list = el.ionic_radii
        if not ir_list:
            return 0.0
        vi = [ir for ir in ir_list if str(getattr(ir, 'coordination', '')).upper() == 'VI']
        chosen = vi if vi else ir_list
        r = getattr(chosen[0], 'ionic_radius', None)
        return _safe_get(r)  # mendeleev 已是 pm，无需换算
    except Exception:
        return 0.0


def _get_block_idx(el):
    """block -> index: s=0, p=1, d=2, f=3"""
    try:
        b = str(getattr(el, 'block', 's')).lower().strip()
        return {'s': 0, 'p': 1, 'd': 2, 'f': 3}.get(b, 0)
    except Exception:
        return 0


def _is_metal(el):
    try:
        block = str(getattr(el, 'block', '')).lower()
        z = int(el.atomic_number)
        if block in ('d', 'f'):
            return True
        if block == 's' and z not in (1, 2):
            return True
        if block == 'p' and z in P_BLOCK_METALS:
            return True
        return False
    except Exception:
        return False


def _is_transition_metal(el):
    try:
        return str(getattr(el, 'block', '')).lower() == 'd'
    except Exception:
        return False


class AtomPropertyEncoderV2(nn.Module):
    """
    重设计版原子特征编码器（V2）。
    物理特征集：极化率、电离能、化学硬度、d 电子数、离子半径等。
    输入：atomic_numbers [N]，tags [N]（可选）
    输出：[N, out_dim=128]，与 AtomPropertyEncoder 接口完全兼容。
    """

    def __init__(self, max_Z: int = 100, out_dim: int = 128):
        super().__init__()
        self.max_Z = int(max_Z)
        self.out_dim = int(out_dim)

        NUM_FEATS = 12   # 数值特征数量
        NUM_FLAGS = 3    # 二值标志数量

        block_table = []
        num_table = []
        flags_table = []

        try:
            from mendeleev import element as _element
            for Z in range(0, self.max_Z + 1):
                if Z == 0:
                    block_table.append(0)
                    num_table.append([0.0] * NUM_FEATS)
                    flags_table.append([0.0] * NUM_FLAGS)
                    continue

                el = _element(Z)

                # ── 类别 A：block ──────────────────────────────────────────
                block_table.append(_get_block_idx(el))

                # ── 类别 B：12 个物理数值特征 ─────────────────────────────
                ie1 = _get_ie(el, 1)
                ie2 = _get_ie(el, 2)
                ea  = _safe_get(getattr(el, 'electron_affinity', None))
                hardness = (ie1 - ea) / 2.0 if ie1 > 0 else 0.0

                polar    = _safe_get(getattr(el, 'dipole_polarizability', None))
                en_p     = _safe_get(getattr(el, 'en_pauling', None))

                # 共价半径：优先 Pyykkö，再回退通用属性
                cov_r = _safe_get(getattr(el, 'covalent_radius_pyykko', None))
                if cov_r == 0.0:
                    cov_r = _safe_get(getattr(el, 'covalent_radius', None))

                vdw_r   = _safe_get(getattr(el, 'vdw_radius', None))
                ionic_r = _get_ionic_radius_VI(el)
                d_elec  = float(VALENCE_D_ELECTRONS.get(Z, 0))
                en_a    = _safe_get(getattr(el, 'electronegativity_allen', None))
                nval    = _safe_get(getattr(el, 'nvalence', None))

                num_table.append([
                    ie1, ie2, polar, hardness, en_p,
                    cov_r, vdw_r, ionic_r, d_elec,
                    ea, en_a, nval,
                ])

                # ── 类别 C：3 个二值标志 ──────────────────────────────────
                flags_table.append([
                    float(_is_metal(el)),
                    float(_is_transition_metal(el)),
                    float(Z in OMS_METALS),
                ])

        except Exception as e:
            import logging
            logging.warning(f"AtomPropertyEncoderV2: mendeleev load failed ({e}), using zeros")
            block_table = [0] * (self.max_Z + 1)
            num_table   = [[0.0] * NUM_FEATS for _ in range(self.max_Z + 1)]
            flags_table = [[0.0] * NUM_FLAGS for _ in range(self.max_Z + 1)]

        # ── 注册 buffer ────────────────────────────────────────────────────
        self.register_buffer("block_index",
                             torch.tensor(block_table, dtype=torch.long))

        num_t = torch.tensor(num_table, dtype=torch.float32)
        # z-score 标准化（逐特征）
        with torch.no_grad():
            mean = num_t.mean(dim=0, keepdim=True)
            std  = num_t.std(dim=0, keepdim=True).clamp_min(1e-6)
            num_t = (num_t - mean) / std
        self.register_buffer("num_values", num_t)

        flags_t = torch.tensor(flags_table, dtype=torch.float32)
        self.register_buffer("flag_values", flags_t)

        # ── 嵌入层 ────────────────────────────────────────────────────────
        # Z Embedding：0..max_Z
        self.z_embedding = nn.Embedding(self.max_Z + 1, 32, padding_idx=0)
        # Block Embedding：s/p/d/f → 0/1/2/3（5 类保留余量）
        self.block_embedding = nn.Embedding(5, 16, padding_idx=0)
        # Tag Embedding：0/1/2
        self.tag_embedding = nn.Embedding(3, 16)

        # ── 投影 MLP：79 → 256 → 128 + LayerNorm ─────────────────────────
        # 32 + 16 + 16 + 12 + 3 = 79
        in_dim = 32 + 16 + 16 + NUM_FEATS + NUM_FLAGS
        self.project = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self,
                atomic_numbers: torch.Tensor,
                tags: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            atomic_numbers: [N] LongTensor，原子序数
            tags:           [N] LongTensor，0=表面/1=体相/2=虚原子（可选）
        Returns:
            [N, out_dim=128]
        """
        idx = atomic_numbers.clamp(0, self.max_Z)

        # 学习型嵌入
        z_emb = self.z_embedding(idx)                        # [N, 32]
        b_idx = self.block_index[idx]
        b_emb = self.block_embedding(b_idx)                  # [N, 16]

        if tags is not None:
            t_emb = self.tag_embedding(tags.long())          # [N, 16]
        else:
            t_emb = torch.zeros(idx.size(0), 16,
                                device=idx.device, dtype=z_emb.dtype)

        # 物理数值 & 标志
        phys  = self.num_values[idx]                         # [N, 12]
        flags = self.flag_values[idx]                        # [N,  3]

        combined = torch.cat([z_emb, b_emb, t_emb, phys, flags], dim=-1)  # [N, 79]
        return self.project(combined)                        # [N, out_dim]
