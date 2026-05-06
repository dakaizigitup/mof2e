#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""filter_keep_lowest_variant_and_strip_adsorbates.py

【新需求（更严格）：一个 MOF 只保留一个输出样本】

输入 LMDB 中同一个 MOF（例如 name 以 "DAWDAI_0.05_1" 开头）可能包含多种吸附体系与多种位置：
  - DAWDAI_0.05_1_w_CO2_1
  - DAWDAI_0.05_1_w_CO2_random_1
  - DAWDAI_0.05_1_w_CO2_H2O_5
  - DAWDAI_0.05_1_w_H2O_1

要求：
1) 输出 LMDB：每个 MOF 仅输出 1 条样本。
2) 输出样本的 name 只保留 MOF 基名（即去掉 "_w" 以及后面所有内容），例如："DAWDAI_0.05_1"。
3) 输出样本的 y_relaxed 改成“字典/映射”：
     y_relaxed[(n, m)] = energy
   其中 n=H2O 个数，m=CO2 个数，value 为对应最低能量。
   例如：
     (0,0) ->（若存在无吸附物样本）
     (0,1) -> CO2 的最低能量（在 CO2_1 与 CO2_random_1 等中取最低）
     (1,1) -> CO2+H2O 的最低能量
     (1,0) -> H2O 的最低能量
4) 结构仅保留 MOF 框架：删除 tags==2 的吸附物原子，并清理与吸附物强相关字段。

实现说明：
- 以 name 中 "_w_" 为分隔：
    base = name.split('_w_', 1)[0]
    tail = name.split('_w_', 1)[1]
- 解析 tail 中的吸附物计数：
    CO2: 识别 "CO2" 或误写 "C02"（字母 O/数字 0 混用）
    H2O: 识别 "H2O" 或误写 "H20"
  计数规则：只看是否出现（当前数据看起来都是 0/1），出现则记 1，否则 0。
- variant 去重：同一 base、同一 (n,m) 可能有多个样本（如 CO2_1 vs CO2_random_1），取能量最低。

注意：
- y_relaxed 字典的 key 使用 tuple (n,m)。LMDB 里通过 pickle 存储，一般可直接写入。
  如果你后续训练/读取代码不接受 tuple-key dict，需要我改成字符串 key（例如 "n=1,m=0"）。

"""

import math
import os
import pickle
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lmdb
from tqdm import tqdm

from fairchem.core.datasets import LmdbDataset


# ================== 配置区（按需修改） ==================
SRC_LMDB = "/home/dell/autodl-tmp/lorafair/data/is2r/filtered_val2"
DST_LMDB = "/home/dell/autodl-tmp/lorafair/data/is2r/newODAC/mof_val.lmdb"
ENERGY_KEY = "y_relaxed"
ADSORBATE_TAG_VALUE = 2
# =======================================================


def build_dataset(src_path: str) -> LmdbDataset:
    return LmdbDataset({"src": src_path, "normalize_labels": False})


def _to_scalar(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        pass
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _to_scalar(x[0])
    return None


def extract_energy(sample: Any, energy_key: str) -> Optional[float]:
    if isinstance(sample, dict):
        return _to_scalar(sample.get(energy_key))
    if hasattr(sample, energy_key):
        return _to_scalar(getattr(sample, energy_key))
    return None


def get_name(sample: Any) -> Optional[str]:
    if isinstance(sample, dict):
        nm = sample.get("name")
    else:
        nm = getattr(sample, "name", None)
    return None if nm is None else str(nm)


def split_base_and_tail(name: str) -> Tuple[str, str]:
    """返回 (base, tail)。若没有 _w_，tail 为空字符串。"""
    if "_w_" not in name:
        return name, ""
    base, tail = name.split("_w_", 1)
    return base, tail


_token_re = re.compile(r"[A-Za-z0-9]+")


def parse_counts_from_tail(tail: str) -> Tuple[int, int]:
    """从 name 的 _w_ 后缀解析 (n_H2O, m_CO2)。

    规则（按你的最新说明）：
    - 例如："CO2_2H2O_3" 表示 1 个 CO2、2 个 H2O；末尾 "_3" 是位置编号，不计入数量。
    - 数量由物种前面的数字决定：
        * 没有数字前缀 -> 1（如 CO2、H2O）
        * 有数字前缀  -> 该数字（如 2H2O、3CO2）
    - 兼容误写：C02/H20。

    注意：tail 里可能还有 random、site 等其它 token，这些会被忽略。
    """
    if not tail:
        return 0, 0

    # tail 常见形态："CO2_2H2O_3" / "CO2_random_1" / "H2O_1" 等
    parts = [p for p in tail.split("_") if p]

    n_h2o = 0
    m_co2 = 0

    # 匹配：可选数字前缀 + 物种名（CO2/H2O 及其误写）
    species_re = re.compile(r"^(?P<count>\d+)?(?P<spec>C(?:O|0)2|H2O|H20)$", re.IGNORECASE)

    for p in parts:
        # 末尾位置编号通常是纯数字：_3/_1/... 直接忽略
        if p.isdigit():
            continue

        mm = species_re.match(p)
        if not mm:
            # 例如 random/siteA 等其它词，忽略
            continue

        cnt = int(mm.group("count")) if mm.group("count") else 1
        spec = mm.group("spec").upper()

        if spec in ("H2O", "H20"):
            n_h2o += cnt
        elif spec in ("CO2", "C02"):
            m_co2 += cnt

    return n_h2o, m_co2


def _is_tensor(x) -> bool:
    try:
        import torch

        return torch.is_tensor(x)
    except Exception:
        return False


def _index_select(x, keep_idx):
    if x is None:
        return None

    if _is_tensor(x):
        import torch

        keep_idx = torch.as_tensor(keep_idx, dtype=torch.long, device=x.device)
        return x.index_select(0, keep_idx)

    if "numpy" in str(type(x)):
        try:
            import numpy as np

            return x[np.asarray(keep_idx, dtype=np.int64)]
        except Exception:
            return x

    if isinstance(x, (list, tuple)):
        kept = [x[int(i)] for i in keep_idx]
        return type(x)(kept) if isinstance(x, tuple) else kept

    return x


def strip_adsorbates(sample: Any) -> Any:
    """删除吸附物原子（tags==ADSORBATE_TAG_VALUE）并尽量清理相关字段。"""
    import copy

    s = copy.copy(sample)

    tags = sample.get("tags") if isinstance(sample, dict) else getattr(sample, "tags", None)
    if tags is None:
        return s

    if _is_tensor(tags):
        import torch

        keep_mask = tags != ADSORBATE_TAG_VALUE
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).view(-1)
        keep_idx_list = keep_idx.detach().cpu().tolist()
    else:
        keep_idx_list = [i for i, t in enumerate(tags) if int(t) != ADSORBATE_TAG_VALUE]

    atom_fields = [
        "pos",
        "positions",
        "atomic_numbers",
        "atoms",
        "tags",
        "fixed",
        "force",
        "forces",
        "charges",
        "magmoms",
        "velocities",
        "node_features",
    ]

    def _get(field):
        return sample.get(field) if isinstance(sample, dict) else getattr(sample, field, None)

    def _set(field, val):
        if isinstance(s, dict):
            if val is None:
                s.pop(field, None)
            else:
                s[field] = val
        else:
            if val is None:
                if hasattr(s, field):
                    try:
                        delattr(s, field)
                    except Exception:
                        pass
            else:
                setattr(s, field, val)

    for f in atom_fields:
        x = _get(f)
        if x is None:
            continue
        try:
            _set(f, _index_select(x, keep_idx_list))
        except Exception:
            pass

    # natoms 更新
    if isinstance(sample, dict):
        if "natoms" in sample:
            s["natoms"] = len(keep_idx_list)
    else:
        if hasattr(sample, "natoms"):
            try:
                setattr(s, "natoms", len(keep_idx_list))
            except Exception:
                pass

    # 删掉一些“显式吸附物描述字段”（不删能量字典 y_relaxed）
    drop_fields = [
        "adsorbate",
        "adsorbate_atoms",
        "ads_symbols",
        "adsorbate_info",
        "adsorption_energy",
        "ads_energy",
        "e_ads",
        "E_ads",
    ]
    for f in drop_fields:
        if isinstance(s, dict):
            s.pop(f, None)
        else:
            if hasattr(s, f):
                try:
                    delattr(s, f)
                except Exception:
                    pass

    return s


def set_field(sample: Any, key: str, value: Any) -> None:
    if isinstance(sample, dict):
        sample[key] = value
    else:
        setattr(sample, key, value)


def delete_field(sample: Any, key: str) -> None:
    if isinstance(sample, dict):
        sample.pop(key, None)
    else:
        if hasattr(sample, key):
            try:
                delattr(sample, key)
            except Exception:
                pass


def write_lmdb(values: Iterable[bytes], dst_lmdb: str) -> None:
    values = list(values)
    if not values:
        raise ValueError("没有可写入的数据")

    os.makedirs(os.path.dirname(os.path.abspath(dst_lmdb)), exist_ok=True)

    total_bytes = sum(len(v) for v in values)
    map_size = int(total_bytes * 1.3) + 64 * 1024 * 1024

    env = lmdb.open(dst_lmdb, map_size=map_size, subdir=False)
    try:
        with env.begin(write=True) as txn:
            for i, v in enumerate(tqdm(values, desc="写入输出 LMDB", ncols=100)):
                txn.put(str(i).encode(), v)
        env.sync()
    finally:
        env.close()


def main():
    src = SRC_LMDB
    dst = DST_LMDB

    print("配置：")
    print(f"SRC              = {src}")
    print(f"DST              = {dst}")
    print(f"ENERGY_KEY        = {ENERGY_KEY}")
    print(f"ADS_TAG_VALUE     = {ADSORBATE_TAG_VALUE}\n")

    dataset = build_dataset(src)
    n_total = len(dataset)
    print(f"输入数据集大小：{n_total}")

    # best_energy[base][(n,m)] = (energy, dataset_idx, full_name)
    best_energy: Dict[str, Dict[Tuple[int, int], Tuple[float, int, str]]] = defaultdict(dict)

    skipped_no_name = 0
    skipped_no_energy = 0

    # 统计每种 (n,m) 出现次数、以及每种 (n,m) 去重后保留次数
    seen_nm_counter = Counter()
    kept_nm_counter = Counter()

    for i in tqdm(range(n_total), desc="扫描并聚合到 MOF", ncols=100):
        s = dataset[i]
        full_name = get_name(s)
        if not full_name:
            skipped_no_name += 1
            continue

        e = extract_energy(s, ENERGY_KEY)
        if e is None or (isinstance(e, float) and (math.isnan(e) or math.isinf(e))):
            skipped_no_energy += 1
            continue
        e = float(e)

        base, tail = split_base_and_tail(full_name)
        nm = parse_counts_from_tail(tail)  # (n_h2o, m_co2)
        seen_nm_counter[nm] += 1

        cur = best_energy[base].get(nm)
        # 同一 base + 同一 (n,m)：只保留最低能量（用于 CO2_1 vs CO2_random_1 这种）
        if cur is None or e < cur[0]:
            best_energy[base][nm] = (e, i, full_name)

    # 统计 kept_nm
    for base, mp in best_energy.items():
        for nm in mp.keys():
            kept_nm_counter[nm] += 1

    mof_count = len(best_energy)

    print("\n扫描统计：")
    print(f"  输入样本数            : {n_total}")
    print(f"  MOF 数（输出条数）     : {mof_count}")
    print(f"  跳过（无 name）       : {skipped_no_name}")
    print(f"  跳过（无能量）       : {skipped_no_energy}")

    # 打印 (n,m) 覆盖情况
    all_nm = sorted(set(seen_nm_counter.keys()) | set(kept_nm_counter.keys()))
    print("\n(n_H2O, m_CO2) 覆盖统计：")
    for nm in all_nm:
        print(f"  {nm}: seen={seen_nm_counter[nm]} kept_best={kept_nm_counter[nm]}")

    # 输出前几个 MOF 示例：展示 y_relaxed 字典键以及它们来自哪个原始 name
    print("\n示例（前 10 个 MOF）：")
    for k, (base, mp) in enumerate(list(best_energy.items())[:10]):
        keys_sorted = sorted(mp.keys())
        print(f"  [{k:02d}] {base}  keys={keys_sorted}")
        for nm in keys_sorted:
            e, idx, full_name = mp[nm]
            print(f"       nm={nm} E={e:.6f} idx={idx} from='{full_name}'")

    # 构造输出：每个 base 只写 1 条样本
    packed_values: List[bytes] = []
    for base, mp in tqdm(best_energy.items(), desc="构造输出样本（MOF-only + 聚合能量字典）", ncols=100):
        # 选一个代表样本来拿结构（只要框架一致即可）
        # 优先用 (0,0) 若存在，否则用能量最小的那个
        if (0, 0) in mp:
            rep_idx = mp[(0, 0)][1]
        else:
            rep_idx = min(mp.values(), key=lambda t: t[0])[1]

        rep_sample = dataset[rep_idx]
        rep_sample = strip_adsorbates(rep_sample)

        # name 改为 base
        set_field(rep_sample, "name", base)

        # y_relaxed 改为 dict[(n,m)] = energy
        y_map: Dict[Tuple[int, int], float] = {nm: info[0] for nm, info in mp.items()}
        set_field(rep_sample, "y_relaxed", y_map)

        # 如果还存在旧的 y（可选：你想保留/删除？这里删除避免混淆）
        delete_field(rep_sample, "y")

        packed_values.append(pickle.dumps(rep_sample, protocol=pickle.HIGHEST_PROTOCOL))

    print(f"\n写入输出 LMDB：{dst}")
    write_lmdb(packed_values, dst)
    print("✅ 完成")


if __name__ == "__main__":
    main()
