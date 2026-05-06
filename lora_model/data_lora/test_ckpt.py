from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
import torch
from fairchem.core import OCPCalculator
from fairchem.core.models.equiformer_v2.equiformer_v2_deprecated import EquiformerV2
model = EquiformerV2()
print(model)


# import torch

# # 加载模型文件
# stat = torch.load("/data0/wfz/code/fairchem/lora_model/data_lora/pretrained_models/eqv2_31M.pt")

# # 获取state_dict的所有键
# keys = stat['state_dict'].keys()

# # 将keys保存到txt文件
# with open('state_dict_keys.txt', 'w') as f:
#     for key in keys:
#         f.write(key + '\n')

# print("Keys已保存到 state_dict_keys.txt 文件")
# # Set up your system as an ASE atoms object
# slab = fcc100("Cu", (3, 3, 3), vacuum=8)
# adsorbate = molecule("CO")
# add_adsorbate(slab, adsorbate, 2.0, "bridge")

# calc = OCPCalculator(
#     model_name="EquiformerV2-31M-S2EF-OC20-All+MD",
#     local_cache="pretrained_models",
#     cpu=False,
# )
# slab.calc = calc

# # Set up LBFGS dynamics object
# dyn = LBFGS(slab)
# dyn.run(0.05, 100)