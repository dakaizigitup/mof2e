from fairchem.core.datasets import LmdbDataset
from fairchem.core.common.utils import setup_logging
from torch.utils.data import DataLoader
import torch

# 3D 可视化（导出 .xyz 并生成一个可直接打开的 HTML）
# 注意：不依赖 notebook；用浏览器打开输出的 html 即可交互旋转缩放
import os
from collections import Counter
import re





# from fairchem.core.datasets import LmdbDataset

# dataset = LmdbDataset(config=dict(src="path_to_ODAC23", r_energy=True, r_forces=True))

# # print a datapoint, you can also use a torch DataLoader to loop through batches
# print(dataset[0])
# 设置日志
setup_logging()

def _safe_int_atomic_numbers(x: torch.Tensor) -> torch.Tensor:
    """atomic_numbers 有时会以 float 存储，这里尽量转回 long 的原子序数。"""
    if not torch.is_tensor(x):
        return x
    if x.dtype.is_floating_point:
        # 常见是 6.0、8.0 这种
        return x.round().to(torch.long)
    return x.to(torch.long)


def _atomic_num_to_symbol(z: int) -> str:
    # 覆盖常见元素；没覆盖的用 Z{num}
    element_map = {
        1: 'H', 2: 'He',
        6: 'C', 7: 'N', 8: 'O', 9: 'F',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
        19: 'K', 20: 'Ca',
        22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        35: 'Br',
        46: 'Pd', 47: 'Ag',
        78: 'Pt', 79: 'Au',
    }
    return element_map.get(int(z), f"Z{int(z)}")


def _export_xyz(path: str, pos: torch.Tensor, atomic_numbers: torch.Tensor) -> None:
    """导出 XYZ：第一行原子数，第二行注释，后面每行：元素 x y z"""
    pos = pos.detach().cpu().float()
    atomic_numbers = _safe_int_atomic_numbers(atomic_numbers).detach().cpu()
    n = pos.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        f.write("generated_by_read_lmdb.py\n")
        for i in range(n):
            sym = _atomic_num_to_symbol(int(atomic_numbers[i].item()))
            x, y, z = pos[i].tolist()
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")


def _make_3dmol_single_html(html_path: str, title: str, xyz_text: str, ads_serials: list, ads_colorscheme: str = 'Jmol') -> None:
    """生成单个结构的 HTML（一个 viewer）。"""
    # 说明：3Dmol.js 从 CDN 加载；如果你机器无外网，需要我再改成本地 js 或改成 plotly 离线
    xyz_text = xyz_text.replace('`', '\\`')
    html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>{title}</title>
  <script src='https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js'></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .title {{ font-weight: 700; margin: 0 0 12px; }}
    #viewer {{ width: 720px; height: 720px; border: 1px solid #ddd; border-radius: 8px; }}
    .hint {{ color: #666; font-size: 12px; margin-top: 8px; }}
  </style>
</head>
<body>
  <div class='title'>{title}</div>
  <div id='viewer'></div>
  <div class='hint'>框架=灰色；吸附物=彩色（按元素 Jmol 配色）。</div>
  <script>
    var viewer = $3Dmol.createViewer(document.getElementById('viewer'), {{backgroundColor: 'white'}});
    viewer.addModel(`{xyz_text}`, 'xyz');
    viewer.setStyle({{}}, {{sphere: {{radius: 0.35, color: '#909090'}}, stick: {{radius: 0.15, color: '#909090'}}}});
    viewer.setStyle({{serial: {ads_serials}}}, {{sphere: {{radius: 0.45, colorscheme: '{ads_colorscheme}'}}, stick: {{radius: 0.20, colorscheme: '{ads_colorscheme}'}}}});
    viewer.zoomTo();
    viewer.render();
  </script>
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


def _visualize_single_mof(dataset, out_dir: str, target_name: str = None, target_index: int = 0):
    """可视化聚合后的单个 MOF（name 不含 _w_ 也能用）。

    目的：确认聚合后结构里已经没有吸附物（tags==2 的原子数应为 0）。

    - target_name: 指定要找的 name（完全匹配或前缀匹配）
    - target_index: 若没给 target_name，则直接用这个索引
    """
    os.makedirs(out_dir, exist_ok=True)

    pick_idx = None
    pick_name = None

    if target_name:
        # 允许完全匹配或前缀匹配
        for i in range(len(dataset)):
            s = dataset[i]
            nm = s.get('name', None) if isinstance(s, dict) else getattr(s, 'name', None)
            if nm is None:
                continue
            nm = str(nm)
            if nm == target_name or nm.startswith(target_name):
                pick_idx = i
                pick_name = nm
                break
        if pick_idx is None:
            print(f"❌ 没找到 target_name={target_name} 的样本。")
            return
    else:
        pick_idx = int(target_index)
        if pick_idx < 0 or pick_idx >= len(dataset):
            print(f"❌ target_index 越界：{pick_idx}，dataset size={len(dataset)}")
            return
        s = dataset[pick_idx]
        pick_name = s.get('name', None) if isinstance(s, dict) else getattr(s, 'name', None)
        pick_name = str(pick_name) if pick_name is not None else f"idx={pick_idx}"

    s = dataset[pick_idx]

    # 兼容 Data 或 dict
    if isinstance(s, dict):
        pos = s['pos']
        atomic_numbers = s['atomic_numbers']
        tags = s.get('tags', None)
        y_relaxed = s.get('y_relaxed', None)
    else:
        pos = s.pos
        atomic_numbers = s.atomic_numbers
        tags = getattr(s, 'tags', None)
        y_relaxed = getattr(s, 'y_relaxed', None)

    atomic_numbers_int = _safe_int_atomic_numbers(atomic_numbers)

    # 统计 tags==2（吸附物）数量
    ads_n = None
    if tags is None:
        print(f"⚠️ 样本 {pick_name} 没有 tags 字段，无法严格确认是否还有吸附物。")
    else:
        if not torch.is_tensor(tags):
            tags = torch.tensor(tags)
        ads_n = int((tags == 2).sum().item())

    print(f"\n🎯 可视化样本：idx={pick_idx} name={pick_name}")
    if ads_n is not None:
        print(f"  tags==2（吸附物）原子数: {ads_n}")
    if y_relaxed is not None:
        # 聚合后 y_relaxed 可能是 dict
        if isinstance(y_relaxed, dict):
            keys = sorted(list(y_relaxed.keys()))
            print(f"  y_relaxed(dict) keys 数量: {len(keys)}")
            print(f"  y_relaxed(dict) keys 示例: {keys[:10]}")
        else:
            print(f"  y_relaxed: {y_relaxed}")

    # 导出 xyz（框架全部灰色显示即可；若 tags==2 仍存在，你也能在统计里看到）
    xyz_path = os.path.join(out_dir, f"MOF__{re.sub(r'[^A-Za-z0-9_.-]+', '_', pick_name)}.xyz")
    _export_xyz(xyz_path, pos, atomic_numbers_int)

    with open(xyz_path, 'r', encoding='utf-8') as f:
        xyz_text = f.read().replace('`', '\\`')

    html_dir = os.path.join(out_dir, "html_each")
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(html_dir, f"VIS__{re.sub(r'[^A-Za-z0-9_.-]+', '_', pick_name)}.html")

    # 聚合后我们期望没有吸附物，所以 ads_serials 传空即可
    _make_3dmol_single_html(
        html_path=html_path,
        title=f"{pick_name} (idx={pick_idx})",
        xyz_text=xyz_text,
        ads_serials=[],
        ads_colorscheme='Jmol',
    )

    print(f"\n✅ 已生成 1 个结构可视化")
    print(f"   XYZ:  {xyz_path}")
    print(f"   HTML: {html_path}")


def _pick_group_and_visualize(dataset, out_dir: str, target_prefix: str = None, max_variants: int = 12):
    """从 dataset 中找一组相同 MOF+吸附物组合但不同编号的样本，导出 xyz 并生成 html。

    注意：这个函数适用于 name 仍然包含 _w_ 的“未聚合数据集”。
    对于聚合后的数据集（name 仅 MOF 基名），请使用 _visualize_single_mof。
    """
    os.makedirs(out_dir, exist_ok=True)

    name_re = re.compile(r"^(?P<mof>[^_]+).*?_w_(?P<ads>.+)_(?P<idx>\d+)$")

    groups = {}  # key=(mof, ads) -> list[(i, name)]
    for i in range(len(dataset)):
        s = dataset[i]
        nm = None
        if isinstance(s, dict):
            nm = s.get('name', None)
        else:
            nm = getattr(s, 'name', None)
        if not nm:
            continue
        m = name_re.match(str(nm))
        if not m:
            continue
        mof = m.group('mof')
        ads = m.group('ads')

        if target_prefix is not None:
            if str(nm).startswith(target_prefix) or mof == target_prefix:
                pass
            else:
                continue

        groups.setdefault((mof, ads), []).append((i, str(nm)))

    if not groups:
        print(f"❌ 没找到任何可视化组。你给的 target_prefix={target_prefix}")
        return

    (mof, ads), items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)[0]
    items = sorted(items, key=lambda t: int(t[1].split('_')[-1]) if t[1].split('_')[-1].isdigit() else 0)
    items = items[:max_variants]

    print(f"\n🎯 选中可视化组: mof={mof}, ads={ads}, 样本数={len(items)}")
    for idx, nm in items:
        print(f"  - [{idx}] {nm}")

    views = []
    for k, (ds_idx, nm) in enumerate(items):
        s = dataset[ds_idx]
        if isinstance(s, dict):
            pos = s['pos']
            atomic_numbers = s['atomic_numbers']
            tags = s.get('tags', None)
        else:
            pos = s.pos
            atomic_numbers = s.atomic_numbers
            tags = getattr(s, 'tags', None)

        atomic_numbers_int = _safe_int_atomic_numbers(atomic_numbers)

        if tags is None:
            print(f"⚠️ 样本 {nm} 没有 tags 字段，无法区分框架/吸附物。")
            continue
        if not torch.is_tensor(tags):
            tags = torch.tensor(tags)
        ads_mask = tags == 2

        xyz_path = os.path.join(out_dir, f"{mof}__{ads}__{k+1:02d}.xyz")
        _export_xyz(xyz_path, pos, atomic_numbers_int)

        with open(xyz_path, 'r', encoding='utf-8') as f:
            xyz_text = f.read().replace('`', '\\`')

        ads_serials = [int(i)+1 for i in torch.nonzero(ads_mask, as_tuple=False).view(-1).tolist()]

        views.append({
            'title': nm,
            'xyz': xyz_text,
            'ads_serials': ads_serials,
            'ads_colorscheme': 'Jmol',
        })

    if not views:
        print("❌ 没有可生成的视图（可能缺少 tags）。")
        return

    html_dir = os.path.join(out_dir, "html_each")
    os.makedirs(html_dir, exist_ok=True)

    for v in views:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", v['title'])
        html_path = os.path.join(html_dir, f"VIS__{safe_name}.html")
        _make_3dmol_single_html(
            html_path=html_path,
            title=v['title'],
            xyz_text=v['xyz'],
            ads_serials=v['ads_serials'],
            ads_colorscheme=v.get('ads_colorscheme', 'Jmol'),
        )

    print(f"\n✅ 已生成 {len(views)} 个 3D HTML（每个结构一个）")
    print(f"   XYZ 输出目录: {out_dir}")
    print(f"   HTML 输出目录: {html_dir}")
    print(f"   示例：{os.path.join(html_dir, 'VIS__' + re.sub(r'[^A-Za-z0-9_.-]+', '_', views[0]['title']) + '.html')}")


def load_with_fairchem_dataset():
    """使用 FairChem 的 LmdbDataset 加载数据"""
    
    # 配置参数
    config = {
        "src": "/home/dell/autodl-tmp/lorafair/data/is2r/mof_only_aggregate_yrelaxed.lmdb",  # LMDB 文件所在目录
        "normalize_labels": False,  # 是否标准化标签
        "target_mean": 0.0,
        "target_std": 1.0,
        "lin_ref": None,  # 线性参考能量
    }
    
    try:
        # 创建数据集
        dataset = LmdbDataset(config)
        print(f"✅ 数据集加载成功")
        print(f"数据集大小: {len(dataset)}")
        
        # === 3D 可视化：选一组同 MOF+吸附物但不同编号的样本 ===
        # 你可以把 target_prefix 改成你关心的组：
        # 例："IJUTEM_0.04_0_w_CO2_H2O" 或只写 "IJUTEM" 也可以
        # ✅ 聚合后数据集（name 不含 _w_）请用这个：直接可视化某一个 MOF
        # 你可以改 target_name 为你想看的 MOF（支持完全匹配或前缀匹配），或改 target_index。
        _visualize_single_mof(
            dataset,
            out_dir=os.path.join(os.path.dirname(__file__), "viz_out"),
            target_name="COQGAS_0.08_0",
            target_index=0,
        )

        # 如果你要看“未聚合数据集（name 含 _w_）的一组不同位置”，再用下面这个：
        # _pick_group_and_visualize(
        #     dataset,
        #     out_dir=os.path.join(os.path.dirname(__file__), "viz_out"),
        #     target_prefix="IJUTEM_0.04_0_w_CO2_H2O",
        #     max_variants=50,
        # )

        
        # 获取第一个样本（详细信息）
        sample = dataset[1]
        print(f"\n样本信息 (索引 1):")
        print(f"类型: {type(sample)}")
        
        # 打印样本属性（原始 key 列表 + 形状）
        if hasattr(sample, '__dict__'):
            keys = list(sample.__dict__.keys())
            print(f"\n🧾 样本包含的所有字段名（共 {len(keys)} 个）:")
            print("  " + ", ".join(sorted(keys)))

            # 只打印“特征类”字段：Tensor 且第一维通常等于原子数 N
            print(f"\n🧬 特征字段（Tensor 且按原子对齐）:")
            n = None
            if hasattr(sample, 'natoms'):
                try:
                    n = int(sample.natoms) if not torch.is_tensor(sample.natoms) else int(sample.natoms.item())
                except Exception:
                    n = None

            for key, value in sample.__dict__.items():
                if not torch.is_tensor(value):
                    continue

                # 过滤掉标量/小张量（例如能量 [1]）
                if value.dim() == 0:
                    continue

                # 如果知道 natoms，就只显示第一维等于 natoms 的张量（典型：pos/atomic_numbers/tags/fixed/...）
                if n is not None:
                    if value.shape[0] != n:
                        continue

                print(f"  {key}: {tuple(value.shape)} ({value.dtype})")

            # 同时打印一些非按原子对齐但也常用的 Tensor（例如 cell [3,3]）
            print(f"\n📦 其他 Tensor 字段（非按原子对齐）:")
            for key, value in sample.__dict__.items():
                if torch.is_tensor(value):
                    if value.dim() == 0:
                        continue
                    if n is not None and value.shape[0] == n:
                        continue
                    print(f"  {key}: {tuple(value.shape)} ({value.dtype})")
        
        # 常见属性
        common_attrs = ['atomic_numbers', 'natoms', 'energy', 'forces', 'cell', 'pbc', 
                        'tags', 'fixed', 'name', 'iid']  # tags 用于标识吸附物/表面原子
        print(f"\n常见属性:")
        for attr in common_attrs:
            if hasattr(sample, attr):
                value = getattr(sample, attr)
                if torch.is_tensor(value):
                    print(f"  {attr}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {attr}: {value}")

        # 打印所有样本的 name（如果存在；避免一次性输出过多，默认最多前 200 条）
        print(f"\n🧾 所有样本的 name（若存在）:")
        max_print = min(200, len(dataset))
        have_name = False
        for i in range(max_print):
            s = dataset[i]
            # LmdbDataset 返回可能是 Data 对象或 dict；两种都兼容
            n = None
            if isinstance(s, dict):
                n = s.get('name', None)
            else:
                if hasattr(s, 'name'):
                    n = getattr(s, 'name')
            if n is not None:
                have_name = True
                print(f"  [{i}] {n}")
        if not have_name:
            print("  （此数据集样本中没有 name 字段）")
        if len(dataset) > max_print:
            print(f"  ... 共 {len(dataset)} 条，仅展示前 {max_print} 条")
        
        # 吸附物信息分析（单条数据）
        print(f"\n🔍 吸附物信息 (索引 1):")
        if hasattr(sample, 'tags'):
            tags = sample.tags
            if torch.is_tensor(tags):
                # tags: 0=表面原子, 1=次表面原子, 2=吸附物原子
                adsorbate_mask = tags == 2
                surface_mask = tags == 0
                subsurface_mask = tags == 1
                
                print(f"  总原子数: {len(tags)}")
                print(f"  吸附物原子数: {adsorbate_mask.sum().item()}")
                print(f"  表面原子数: {surface_mask.sum().item()}")
                print(f"  次表面原子数: {subsurface_mask.sum().item()}")
                
                # 打印吸附物的原子序数
                if adsorbate_mask.sum() > 0 and hasattr(sample, 'atomic_numbers'):
                    adsorbate_atomic_nums = sample.atomic_numbers[adsorbate_mask]
                    print(f"  吸附物原子序数: {adsorbate_atomic_nums.tolist()}")
                    
                    # 转换为元素符号（常见元素）
                    element_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S', 15: 'P'}
                    elements = [element_map.get(num.item(), f'Z{num.item()}') for num in adsorbate_atomic_nums]
                    print(f"  吸附物元素: {elements}")
                    print(f"  吸附物化学式: {''.join(elements)}")
            else:
                print(f"  tags 类型: {type(tags)}, 值: {tags}")
        else:
            print(f"  ⚠️ 样本中没有 'tags' 属性，无法识别吸附物")
        
        # 检查其他可能包含吸附物信息的属性
        adsorbate_attrs = ['adsorbate', 'adsorbate_atoms', 'ads_symbols', 'adsorbate_info']
        for attr in adsorbate_attrs:
            if hasattr(sample, attr):
                value = getattr(sample, attr)
                print(f"  {attr}: {value}")
        
        # 统计整个数据集的吸附物类型[object Object]0 条数据):")
        from collections import Counter
        import tqdm
        
        adsorbate_counter = Counter()
        element_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S', 15: 'P'}
        
        # 设置遍历的数据量，避免数据集过大时耗时太长
        num_samples_to_scan =  len(dataset)
        
        for i in tqdm.tqdm(range(num_samples_to_scan), desc="正在统计吸附物"):
            sample_i = dataset[i]
            if hasattr(sample_i, 'tags') and hasattr(sample_i, 'atomic_numbers'):
                tags_i = sample_i.tags
                if torch.is_tensor(tags_i):
                    adsorbate_mask_i = tags_i == 2
                    if adsorbate_mask_i.sum() > 0:
                        adsorbate_nums = sample_i.atomic_numbers[adsorbate_mask_i]
                        # 对原子序数排序，确保化学式唯一 (e.g., HHO -> H2O)
                        sorted_nums = sorted(adsorbate_nums.tolist())
                        elements = [element_map.get(num, f'Z{num}') for num in sorted_nums]
                        
                        # 生成规范的化学式 (e.g., [C, H, H, H] -> CH3)
                        element_counts = Counter(elements)
                        formula = "".join([f"{elem}{count if count > 1 else ''}" for elem, count in sorted(element_counts.items())])
                        
                        adsorbate_counter[formula] += 1
                    else:
                        adsorbate_counter["无吸附物"] += 1
                else:
                    adsorbate_counter["tags 非张量"] += 1
            else:
                adsorbate_counter["缺少 tags/atomic_numbers"] += 1
        
        # 打印统计结果
        print("\n--- 统计结果 ---")
        if adsorbate_counter:
            # 按数量从高到低排序
            sorted_adsorbates = adsorbate_counter.most_common()
            for adsorbate, count in sorted_adsorbates:
                print(f"  {adsorbate}: {count}")
        else:
            print("  未统计到任何吸附物")

        # 检查 y / energy 是否存在负数（常见：吸附能可能为负）
        print("\n🔎 检查标签是否有负数（y / energy / y_relaxed）:")
        neg_count = 0
        pos_count = 0
        nan_count = 0
        total_checked = 0
        min_val = float('inf')
        max_val = float('-inf')

        def _to_float(v):
            if v is None:
                return None
            if torch.is_tensor(v):
                if v.numel() == 0:
                    return None
                v = v.detach().cpu().view(-1)[0].item()
            try:
                return float(v)
            except Exception:
                return None

        for i in tqdm.tqdm(range(num_samples_to_scan), desc="检查y是否为负"):
            s = dataset[i]
            yv = None
            if isinstance(s, dict):
                for k in ['y', 'energy', 'y_relaxed']:
                    if k in s:
                        yv = _to_float(s[k])
                        if yv is not None:
                            break
            else:
                for k in ['y', 'energy', 'y_relaxed']:
                    if hasattr(s, k):
                        yv = _to_float(getattr(s, k))
                        if yv is not None:
                            break

            if yv is None:
                nan_count += 1
                continue

            total_checked += 1
            min_val = min(min_val, yv)
            max_val = max(max_val, yv)
            if yv < 0:
                neg_count += 1
            else:
                pos_count += 1

        print(f"  检查条数: {total_checked} / {num_samples_to_scan}")
        print(f"  负数数量: {neg_count}")
        print(f"  非负数量: {pos_count}")
        print(f"  无法读取数量: {nan_count}")
        if total_checked > 0:
            print(f"  最小值 min: {min_val}")
            print(f"  最大值 max: {max_val}")

        print(f"\n数据集大小: {len(dataset)}")
        return dataset
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

# 使用示例
dataset = load_with_fairchem_dataset()

