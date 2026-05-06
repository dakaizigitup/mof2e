"""
global_emb_v2.py  ―  三模态预训练模型融合版 MOF 全局特征编码器

特征通道及输出维度:
  ① SMILES 三模态融合 (Ligand_1/2/3)          → 256
  ② 孔结构数值特征 (6孔 + has_NASA 二值)      →  64
  ③ 分类特征 (金属/OMS/拓扑/封端配体)         → 128
  ④ Ligand_1 附加特征 (length+struct+groups)  →  64
  ⑤ Metals SMILES (SMI-TED)                   →  32
  最终: [256+64+128+64+32=544] → 256 → 64

缺失率处理决策:
  - NAV_cm3_g (98% 零值)   : 完全删除
  - NASA_m2_cm3 (90% 零值) : 用 has_NASA 二值特征代替原始值
  - Ligand_2/3 附加特征     : 缺失率 75-99%，全部删除，仅保留 Ligand_1
  - structures/groups       : 7%/17% 缺失，用 '<NONE>' 占位 token 处理
  - Terminal Ligands        : 46% 缺失，用 '<NONE>' 处理
  - Topologies              : 47% UNKNOWN，UNKNOWN 作为合法类别直接编码

预训练模型（全部本地离线加载）:
  SMI-TED   : smi_ted_light/ (bert_vocab + .pt)
  SELFIES-TED: selfies_ted/pretrained/
  MHG-GED   : mhg_model/pickles/pytorch_model.bin

使用方法:
  encoder = MOFGlobalEncoderV2(
      excel_path   = "path/to/MOF_embedding.xlsx",
      smi_ted_folder   = "path/to/smi_ted_light",
      selfies_ted_path = "path/to/selfies_ted/pretrained",
      mhg_path         = "path/to/mhg_model/pickles/pytorch_model.bin",
      device       = "cuda",
      pretrained_device = "cpu",  # 可选：预训练大模型放 CPU，轻量可学习层仍可放 GPU
      cache_path   = "path/to/smiles_cache.pt",  # 可选，跨 run 重用预计算结果
  )
  emb = encoder(["MOF1", "MOF2"])  # → Tensor [2, 64]
"""

import os
import sys
import logging
import pickle
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 小模块
# ─────────────────────────────────────────────────────────────────────────────

class IntraModalGating(nn.Module):
    """对单个模态向量做 sigmoid 门控，选择有效维度。"""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class InterModalAttention(nn.Module):
    """对多个模态向量做 softmax 加权融合。"""
    def __init__(self, proj_dim: int):
        super().__init__()
        self.score = nn.Linear(proj_dim, 1)

    def forward(self, modal_vecs: List[torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        modal_vecs : List[Tensor[proj_dim]]  长度 = n_modalities
        mask       : BoolTensor[n_modalities]  True 表示屏蔽（该模态缺失）
        返回       : Tensor[proj_dim]
        """
        stacked = torch.stack(modal_vecs, dim=0)         # [M, proj_dim]
        scores  = self.score(stacked).squeeze(-1)         # [M]
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=0)            # [M]
        return (weights.unsqueeze(-1) * stacked).sum(dim=0)  # [proj_dim]


class LigandAttentionPool(nn.Module):
    """对多个配体向量（可能有缺失）做 softmax attention 池化。"""
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, lig_vecs: List[torch.Tensor],
                lig_mask: List[bool]) -> torch.Tensor:
        """
        lig_vecs : List[Tensor[dim]]  (长度 3，对应 L1/L2/L3)
        lig_mask : List[bool]          True 表示该配体存在
        """
        valid = [(v, m) for v, m in zip(lig_vecs, lig_mask) if m]
        if not valid:
            return torch.zeros_like(lig_vecs[0])
        if len(valid) == 1:
            return valid[0][0]
        stacked = torch.stack([v for v, _ in valid], dim=0)  # [k, dim]
        scores  = self.score(stacked).squeeze(-1)             # [k]
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(-1) * stacked).sum(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# 主编码器
# ─────────────────────────────────────────────────────────────────────────────

class MOFGlobalEncoderV2(nn.Module):

    # 数值列：6 孔结构 + has_NASA 二值 = 7 维
    _NUMERIC_COLS = [
        "LCD(最大腔体直径)", "PLD(最小孔径)", "LFPD(最大自由球体直径)",
        "ASA_m2_cm3(比表面积)", "AV_VF(孔隙率)", "AV_cm3_g(可接触体积)",
    ]
    _LOG1P_COLS = {
        "LCD(最大腔体直径)", "PLD(最小孔径)", "LFPD(最大自由球体直径)",
        "ASA_m2_cm3(比表面积)", "AV_cm3_g(可接触体积)",
    }
    _NASA_COL = "NASA_m2_cm3(不可接触表面积)"   # 转成 has_NASA 二值

    _LIGAND_COLS  = ["Ligand_1", "Ligand_2", "Ligand_3"]
    _METALS_COL   = "Metals(金属节点)"

    def __init__(
        self,
        excel_path:       str,
        smi_ted_folder:   str,
        selfies_ted_path: str,
        mhg_path:         str,
        global_dim:       int   = 64,
        proj_dim:         int   = 256,
        device:           str   = "cuda",
        pretrained_device: Optional[str] = None,
        cache_path:       Optional[str] = None,
    ):
        super().__init__()
        self.excel_path       = excel_path
        self.smi_ted_folder   = smi_ted_folder
        self.selfies_ted_path = selfies_ted_path
        self.mhg_path         = mhg_path
        self.global_dim       = global_dim
        self.proj_dim         = proj_dim
        self.device           = torch.device(device)
        self.pretrained_device = torch.device(pretrained_device or device)
        self.cache_path       = cache_path

        self._warned_missing: set = set()

        # 1. 读 Excel
        self.df = pd.read_excel(excel_path).fillna("")
        self._build_name_index()

        # 2. 加载预训练模型（推理模式，冻结）
        self.smi_ted_model    = None
        self.selfies_model    = None
        self.mhg_model        = None
        self.mhg_dim          = 256   # 默认，加载后更新
        self._load_pretrained_models()

        # 3. 构建 sklearn 编码器 / 词表
        self._build_sklearn_encoders()

        # 4. 构建 PyTorch 子模块
        self._build_networks()

        # 5. 预计算 SMILES 嵌入缓存
        self.smiles_cache: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        self._precompute_smiles_cache()
        self._release_pretrained_models()

        self.to(self.device)

    # ──────────────────────────────────────────────
    # 初始化辅助
    # ──────────────────────────────────────────────

    @staticmethod
    def _norm_name(name: str) -> str:
        return str(name).strip().upper().split("_")[0]

    def _build_name_index(self):
        self.name_to_idx = {
            self._norm_name(n): i for i, n in enumerate(self.df["Name"])
        }

    def _load_pretrained_models(self):
        # ── SMI-TED ──
        try:
            _models_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                "../../../materials-main/models"
            ))
            if _models_dir not in sys.path:
                sys.path.insert(0, _models_dir)
            from smi_ted.smi_ted_light.load import load_smi_ted
            self.smi_ted_model = load_smi_ted(local_folder=self.smi_ted_folder)
            self.smi_ted_model = self.smi_ted_model.to(self.pretrained_device)
            self.smi_ted_model.eval()
            for p in self.smi_ted_model.parameters():
                p.requires_grad_(False)
            logging.info(f"[V2] SMI-TED loaded OK on {self.pretrained_device}")
        except Exception as e:
            logging.warning(f"[V2] SMI-TED load failed: {e}. Using zeros.")

        # ── SELFIES-TED ──
        try:
            from selfies_ted.load import SELFIES
            self.selfies_model = SELFIES()
            self.selfies_model.load(local_path=self.selfies_ted_path)
            self.selfies_model = self.selfies_model.to(self.pretrained_device)
            self.selfies_model.eval()
            for p in self.selfies_model.parameters():
                p.requires_grad_(False)
            logging.info(f"[V2] SELFIES-TED loaded OK on {self.pretrained_device}")
        except Exception as e:
            logging.warning(f"[V2] SELFIES-TED load failed: {e}. Using zeros.")

        # ── MHG-GED ──
        try:
            from mhg_model.load import load as load_mhg
            self.mhg_model = load_mhg(local_path=self.mhg_path)
            self.mhg_model.model = self.mhg_model.model.to(self.pretrained_device)
            self.mhg_model.model.eval()
            for p in self.mhg_model.model.parameters():
                p.requires_grad_(False)
            # 探测输出维度
            _test = self._encode_mhg_single("CCO")
            if _test is not None:
                self.mhg_dim = _test.shape[-1]
            logging.info(f"[V2] MHG-GED loaded OK on {self.pretrained_device}, mhg_dim={self.mhg_dim}")
        except Exception as e:
            logging.warning(f"[V2] MHG-GED load failed: {e}. Using zeros.")

    def _build_sklearn_encoders(self):
        df = self.df

        # ── 数值缩放器 ──
        self.num_scalers: Dict[str, StandardScaler] = {}
        for col in self._NUMERIC_COLS:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values
            if col in self._LOG1P_COLS:
                vals = np.log1p(np.maximum(vals, 0))
            sc = StandardScaler()
            sc.fit(vals.reshape(-1, 1))
            self.num_scalers[col] = sc

        # ── 多标签金属词表 ──
        def _build_vocab(col):
            vocab = set()
            for v in df.get(col, pd.Series(dtype=str)):
                if isinstance(v, str) and v.strip():
                    vocab.update(t.strip() for t in v.split(",") if t.strip())
            return {t: i for i, t in enumerate(sorted(vocab))}

        self.metal_vocab = _build_vocab("All_Metals(金属种类)")
        self.oms_vocab   = _build_vocab("Open_Metal_Sites(缺陷位点)")

        # ── 单标签分类编码器 ──
        def _fit_le(col, extra_classes=()):
            le = LabelEncoder()
            vals = list(df[col].astype(str)) if col in df.columns else []
            vals += list(extra_classes)
            le.fit(vals if vals else ["__dummy__"])
            return le

        self.le_topology  = _fit_le("Topologies(拓扑)")
        self.le_terminal  = _fit_le("Terminal Ligands(封端配体)", ["<NONE>"])
        self.le_struct1   = _fit_le("structures",  ["<NONE>"])
        self.le_groups1   = _fit_le("groups",      ["<NONE>"])

        # 各分类词表大小（用于构建 Embedding）
        self.n_metals    = max(len(self.metal_vocab), 1)
        self.n_oms       = max(len(self.oms_vocab),   1)
        self.n_topo      = len(self.le_topology.classes_)
        self.n_terminal  = len(self.le_terminal.classes_)
        self.n_struct1   = len(self.le_struct1.classes_)
        self.n_groups1   = len(self.le_groups1.classes_)

        # ligand_1 length 缩放器
        self.len1_scaler = None
        if "length" in df.columns:
            vals = pd.to_numeric(df["length"], errors="coerce").fillna(0).values
            sc = MinMaxScaler()
            sc.fit(vals.reshape(-1, 1))
            self.len1_scaler = sc

    @staticmethod
    def _emb_dim(vocab_size: int) -> int:
        return min(50, max(4, (vocab_size + 1) // 2))

    def _build_networks(self):
        proj  = self.proj_dim
        D_mhg = self.mhg_dim

        # ── ① SMILES 三模态 ──
        # 投影层
        self.proj_smited  = nn.Linear(768,    proj)
        self.proj_selfies = nn.Linear(1024,   proj)
        self.proj_mhg     = nn.Linear(D_mhg,  proj)

        # Intra-modal gating (每个模态独立)
        self.gate_smited  = IntraModalGating(proj)
        self.gate_selfies = IntraModalGating(proj)
        self.gate_mhg     = IntraModalGating(proj)

        # Inter-modal attention（3 模态 → 1 向量）
        self.inter_attn = InterModalAttention(proj)

        # 配体 attention pool（最多 3 个配体）
        self.lig_pool = LigandAttentionPool(proj)

        # ── ② 数值通道 ──  7 = 6 孔结构 + has_NASA
        self.numeric_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
        )

        # ── ③ 分类通道 ──
        def e(n):  return self._emb_dim(n)

        self.emb_metals   = nn.Embedding(self.n_metals,   e(self.n_metals))
        self.emb_oms      = nn.Embedding(self.n_oms,      e(self.n_oms))
        self.emb_topo     = nn.Embedding(self.n_topo,     e(self.n_topo))
        self.emb_terminal = nn.Embedding(self.n_terminal, e(self.n_terminal))

        # Has_OMS 直接用 float (1 dim)
        cat_dim = (e(self.n_metals) + 1 + e(self.n_oms) +
                   e(self.n_topo) + e(self.n_terminal))
        self.categorical_encoder = nn.Sequential(
            nn.Linear(cat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
        )

        # ── ④ Ligand_1 附加特征 ──
        self.emb_struct1 = nn.Embedding(self.n_struct1, e(self.n_struct1))
        self.emb_groups1 = nn.Embedding(self.n_groups1, e(self.n_groups1))
        lig_extra_dim = 1 + e(self.n_struct1) + e(self.n_groups1)
        self.lig_extra_encoder = nn.Sequential(
            nn.Linear(lig_extra_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
        )

        # ── ⑤ Metals SMILES (SMI-TED) ──
        self.metals_proj = nn.Linear(768, 32)

        # ── 最终融合 ──
        final_in = proj + 64 + 128 + 64 + 32   # 256+64+128+64+32 = 544
        self.final_encoder = nn.Sequential(
            nn.Linear(final_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.global_dim),
        )

    # ──────────────────────────────────────────────
    # 预训练模型推理（无梯度）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _encode_smited_batch(self, smiles_list: List[str]) -> np.ndarray:
        """返回 numpy [N, 768]，无效项为 NaN 行。"""
        if self.smi_ted_model is None:
            return np.full((len(smiles_list), 768), np.nan)
        try:
            t = self.smi_ted_model.encode(smiles_list, return_torch=True)
            return t.cpu().numpy()
        except Exception as e:
            logging.warning(f"[V2] SMI-TED encode error: {e}")
            return np.full((len(smiles_list), 768), np.nan)

    @torch.no_grad()
    def _encode_selfies_batch(self, smiles_list: List[str]) -> np.ndarray:
        """返回 numpy [N, 1024]。"""
        if self.selfies_model is None:
            return np.full((len(smiles_list), 1024), np.nan)
        try:
            t = self.selfies_model.encode(smiles_list, return_tensor=True)
            return t.cpu().numpy()
        except Exception as e:
            logging.warning(f"[V2] SELFIES-TED encode error: {e}")
            return np.full((len(smiles_list), 1024), np.nan)

    @torch.no_grad()
    def _encode_mhg_single(self, smiles: str) -> Optional[torch.Tensor]:
        if self.mhg_model is None:
            return None
        try:
            out = self.mhg_model.encode([smiles])
            if out and out[0] is not None:
                v = out[0]
                return v.squeeze(0).detach().cpu()
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────────
    # 预计算缓存
    # ──────────────────────────────────────────────

    def _precompute_smiles_cache(self):
        """对 Excel 中所有 MOF，预计算三个模型对各配体的嵌入并缓存。"""
        # 尝试从磁盘加载
        if self.cache_path and os.path.isfile(self.cache_path):
            try:
                self.smiles_cache = torch.load(self.cache_path, map_location="cpu")
                logging.info(f"[V2] SMILES cache loaded from {self.cache_path} "
                             f"({len(self.smiles_cache)} entries)")
                return
            except Exception as e:
                logging.warning(f"[V2] Cache load failed ({e}), recomputing.")

        logging.info("[V2] Pre-computing SMILES embeddings for all MOFs ...")

        # 收集所有需要编码的列
        cols = self._LIGAND_COLS + [self._METALS_COL]

        # 批量收集各列的 SMILES
        col_smiles: Dict[str, List[str]] = {c: [] for c in cols}
        col_valid_idx: Dict[str, List[int]] = {c: [] for c in cols}

        for i, row in self.df.iterrows():
            for c in cols:
                smi = str(row.get(c, "")).strip()
                if smi and smi.lower() != "nan":
                    col_smiles[c].append(smi)
                    col_valid_idx[c].append(i)

        # 对每列分别批量推理
        col_smited_emb:  Dict[str, Dict[int, np.ndarray]] = {}
        col_selfies_emb: Dict[str, Dict[int, np.ndarray]] = {}
        col_mhg_emb:     Dict[str, Dict[int, Optional[torch.Tensor]]] = {}

        for c in cols:
            smis  = col_smiles[c]
            idxs  = col_valid_idx[c]
            if not smis:
                col_smited_emb[c]  = {}
                col_selfies_emb[c] = {}
                col_mhg_emb[c]     = {}
                continue

            logging.info(f"[V2]   Encoding col '{c}' ({len(smis)} entries) ...")

            # SMI-TED (Metals 也需要 SMI-TED)
            arr_st = self._encode_smited_batch(smis)   # [N, 768]
            col_smited_emb[c] = {idx: arr_st[j] for j, idx in enumerate(idxs)}

            # SELFIES-TED (仅配体，Metals 不需要)
            if c in self._LIGAND_COLS:
                arr_sf = self._encode_selfies_batch(smis)  # [N, 1024]
                col_selfies_emb[c] = {idx: arr_sf[j] for j, idx in enumerate(idxs)}

                # MHG-GED（逐条）
                mhg_dict: Dict[int, Optional[torch.Tensor]] = {}
                for j, (smi, idx) in enumerate(zip(smis, idxs)):
                    mhg_dict[idx] = self._encode_mhg_single(smi)
                col_mhg_emb[c] = mhg_dict
            else:
                col_selfies_emb[c] = {}
                col_mhg_emb[c]     = {}

        # 组装 per-MOF 缓存
        for norm_name, df_idx in self.name_to_idx.items():
            entry: Dict[str, Optional[torch.Tensor]] = {}
            for c in cols:
                # SMI-TED
                arr = col_smited_emb.get(c, {}).get(df_idx)
                if arr is not None and not np.any(np.isnan(arr)):
                    entry[f"{c}_smited"] = torch.tensor(arr, dtype=torch.float32)
                else:
                    entry[f"{c}_smited"] = None

                if c in self._LIGAND_COLS:
                    # SELFIES-TED
                    arr_sf = col_selfies_emb.get(c, {}).get(df_idx)
                    if arr_sf is not None and not np.any(np.isnan(arr_sf)):
                        entry[f"{c}_selfies"] = torch.tensor(arr_sf, dtype=torch.float32)
                    else:
                        entry[f"{c}_selfies"] = None

                    # MHG-GED
                    t_mhg = col_mhg_emb.get(c, {}).get(df_idx)
                    entry[f"{c}_mhg"] = t_mhg  # Tensor 或 None

            self.smiles_cache[norm_name] = entry

        # 保存到磁盘
        if self.cache_path:
            try:
                torch.save(self.smiles_cache, self.cache_path)
                logging.info(f"[V2] SMILES cache saved to {self.cache_path}")
            except Exception as e:
                logging.warning(f"[V2] Cache save failed: {e}")

        logging.info(f"[V2] Pre-computation done. {len(self.smiles_cache)} MOFs cached.")

    def _release_pretrained_models(self):
        released = []
        if self.smi_ted_model is not None:
            self.smi_ted_model = None
            released.append("SMI-TED")
        if self.selfies_model is not None:
            self.selfies_model = None
            released.append("SELFIES-TED")
        if self.mhg_model is not None:
            self.mhg_model = None
            released.append("MHG-GED")
        if released:
            logging.info(
                f"[V2] Released pretrained models after cache build: {', '.join(released)}"
            )
            if self.pretrained_device.type == "cuda":
                torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    # 单 MOF 编码（可学习部分）
    # ──────────────────────────────────────────────

    def _safe_get(self, tensor: Optional[torch.Tensor],
                  zero_size: int) -> torch.Tensor:
        if tensor is not None:
            return tensor.to(self.device)
        return torch.zeros(zero_size, device=self.device)

    def _encode_ligand_trimodal(self, cache_entry: Dict,
                                lig_col: str) -> tuple:
        """
        对一个配体（lig_col）做三模态融合。
        返回 (fused: Tensor[proj_dim], present: bool)
        present=False 表示该配体在本行中缺失（三个模型的缓存均为 None）。
        """
        v_st  = self._safe_get(cache_entry.get(f"{lig_col}_smited"),  768)
        v_sf  = self._safe_get(cache_entry.get(f"{lig_col}_selfies"), 1024)
        v_mhg = self._safe_get(cache_entry.get(f"{lig_col}_mhg"),     self.mhg_dim)

        # 判断该配体是否存在（三者均缺失 = 配体本身缺失）
        present = (cache_entry.get(f"{lig_col}_smited")  is not None or
                   cache_entry.get(f"{lig_col}_selfies") is not None or
                   cache_entry.get(f"{lig_col}_mhg")     is not None)

        # 投影
        p_st  = self.proj_smited(v_st)    # [256]
        p_sf  = self.proj_selfies(v_sf)   # [256]
        p_mhg = self.proj_mhg(v_mhg)     # [256]

        # Intra-modal gating
        p_st  = self.gate_smited(p_st)
        p_sf  = self.gate_selfies(p_sf)
        p_mhg = self.gate_mhg(p_mhg)

        # 模态 mask：若缓存为 None 则屏蔽该模态
        modal_mask = torch.tensor([
            cache_entry.get(f"{lig_col}_smited")  is None,
            cache_entry.get(f"{lig_col}_selfies") is None,
            cache_entry.get(f"{lig_col}_mhg")     is None,
        ], device=self.device)

        fused = self.inter_attn([p_st, p_sf, p_mhg], mask=modal_mask)
        return fused, present

    def encode_mof_by_name(self, mof_name: str) -> torch.Tensor:
        cache = self.smiles_cache.get(mof_name, {})
        row   = self.df.iloc[self.name_to_idx[mof_name]]

        # ── ① SMILES 三模态 ──
        lig_fused: List[torch.Tensor] = []
        lig_valid: List[bool]         = []
        for c in self._LIGAND_COLS:
            fused, present = self._encode_ligand_trimodal(cache, c)
            lig_fused.append(fused)
            lig_valid.append(present)

        smiles_vec = self.lig_pool(lig_fused, lig_valid)  # [256]

        # ── ② 数值特征 ──
        num_vals = []
        for col in self._NUMERIC_COLS:
            v = pd.to_numeric(row.get(col, 0), errors="coerce")
            v = 0.0 if pd.isna(v) else float(v)
            if col in self._LOG1P_COLS:
                v = float(np.log1p(max(v, 0.0)))
            sc = self.num_scalers.get(col)
            if sc is not None:
                v = float(sc.transform([[v]])[0][0])
            num_vals.append(v)

        # has_NASA 二值
        nasa_raw = pd.to_numeric(row.get(self._NASA_COL, 0), errors="coerce")
        has_nasa = 0.0 if (pd.isna(nasa_raw) or float(nasa_raw) == 0.0) else 1.0
        num_vals.append(has_nasa)

        num_tensor = torch.tensor(num_vals, dtype=torch.float32, device=self.device)
        num_vec = self.numeric_encoder(num_tensor)  # [64]

        # ── ③ 分类特征 ──
        cat_parts = []

        # All_Metals（多标签 → mean pooling）
        metals_tokens = [t.strip() for t in str(row.get("All_Metals(金属种类)", "")).split(",")
                         if t.strip() and t.strip() in self.metal_vocab]
        if metals_tokens:
            idxs = torch.tensor([self.metal_vocab[t] for t in metals_tokens],
                                 dtype=torch.long, device=self.device)
            cat_parts.append(self.emb_metals(idxs).mean(dim=0))
        else:
            cat_parts.append(torch.zeros(self.emb_metals.embedding_dim, device=self.device))

        # Has_OMS（直接 float）
        has_oms_raw = str(row.get("Has_OMS(是否有缺陷)", "No")).strip().lower()
        has_oms = 1.0 if has_oms_raw in ("yes", "true", "1") else 0.0
        cat_parts.append(torch.tensor([has_oms], dtype=torch.float32, device=self.device))

        # Open_Metal_Sites（多标签 → mean pooling）
        oms_tokens = [t.strip() for t in str(row.get("Open_Metal_Sites(缺陷位点)", "")).split(",")
                      if t.strip() and t.strip() in self.oms_vocab]
        if oms_tokens:
            idxs = torch.tensor([self.oms_vocab[t] for t in oms_tokens],
                                 dtype=torch.long, device=self.device)
            cat_parts.append(self.emb_oms(idxs).mean(dim=0))
        else:
            cat_parts.append(torch.zeros(self.emb_oms.embedding_dim, device=self.device))

        # Topologies
        topo_val = str(row.get("Topologies(拓扑)", "UNKNOWN")).strip()
        try:
            topo_idx = int(self.le_topology.transform([topo_val])[0])
        except Exception:
            topo_idx = 0
        cat_parts.append(self.emb_topo(
            torch.tensor(topo_idx, dtype=torch.long, device=self.device)))

        # Terminal Ligands
        term_val = str(row.get("Terminal Ligands(封端配体)", "")).strip()
        if not term_val or term_val.lower() == "nan":
            term_val = "<NONE>"
        try:
            term_idx = int(self.le_terminal.transform([term_val])[0])
        except Exception:
            term_idx = int(self.le_terminal.transform(["<NONE>"])[0])
        cat_parts.append(self.emb_terminal(
            torch.tensor(term_idx, dtype=torch.long, device=self.device)))

        cat_vec = self.categorical_encoder(
            torch.cat(cat_parts, dim=0))  # [128]

        # ── ④ Ligand_1 附加特征 ──
        # length
        len1_val = pd.to_numeric(row.get("length", 0), errors="coerce")
        len1_val = 0.0 if pd.isna(len1_val) else float(len1_val)
        if self.len1_scaler is not None:
            len1_val = float(self.len1_scaler.transform([[len1_val]])[0][0])
        len1_tensor = torch.tensor([len1_val], dtype=torch.float32, device=self.device)

        # structures
        struct_val = str(row.get("structures", "")).strip()
        if not struct_val or struct_val.lower() == "nan":
            struct_val = "<NONE>"
        try:
            struct_idx = int(self.le_struct1.transform([struct_val])[0])
        except Exception:
            struct_idx = int(self.le_struct1.transform(["<NONE>"])[0])
        struct_emb = self.emb_struct1(
            torch.tensor(struct_idx, dtype=torch.long, device=self.device))

        # groups
        groups_val = str(row.get("groups", "")).strip()
        if not groups_val or groups_val.lower() == "nan":
            groups_val = "<NONE>"
        try:
            groups_idx = int(self.le_groups1.transform([groups_val])[0])
        except Exception:
            groups_idx = int(self.le_groups1.transform(["<NONE>"])[0])
        groups_emb = self.emb_groups1(
            torch.tensor(groups_idx, dtype=torch.long, device=self.device))

        lig_extra_vec = self.lig_extra_encoder(
            torch.cat([len1_tensor, struct_emb, groups_emb], dim=0))  # [64]

        # ── ⑤ Metals SMILES (SMI-TED) ──
        metals_st = self._safe_get(cache.get(f"{self._METALS_COL}_smited"), 768)
        metals_vec = self.metals_proj(metals_st)  # [32]

        # ── 最终融合 ──
        final_in = torch.cat([smiles_vec, num_vec, cat_vec,
                               lig_extra_vec, metals_vec], dim=0)  # [544]
        out = self.final_encoder(final_in)  # [64]
        return out

    # ──────────────────────────────────────────────
    # forward
    # ──────────────────────────────────────────────

    def forward(self, mof_names: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(mof_names, str):
            mof_names = [mof_names]

        embs = []
        for raw in mof_names:
            name = self._norm_name(raw)
            if name in self.name_to_idx:
                emb = self.encode_mof_by_name(name)
            else:
                if name not in self._warned_missing:
                    logging.warning(f"[V2] MOF '{name}' not found in Excel, using zeros.")
                    self._warned_missing.add(name)
                emb = torch.zeros(self.global_dim,
                                  device=next(self.parameters()).device)

            if torch.isnan(emb).any() or torch.isinf(emb).any():
                logging.warning(f"[V2] NaN/Inf in embedding for '{raw}', clamping.")
                emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

            embs.append(emb)

        return torch.stack(embs, dim=0)

    # ──────────────────────────────────────────────
    # 工具
    # ──────────────────────────────────────────────

    def reset_coverage_stats(self):
        self._warned_missing = set()

    def save_cache(self, path: str):
        torch.save(self.smiles_cache, path)
        logging.info(f"[V2] Cache saved to {path}")
