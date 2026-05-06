# 新建文件：core/models/escn/escn_lora.py

"""
eSCN模型 + LoRA微调版本
基于EquiformerV2LoRA的实现模式，适配到eSCN架构
"""

from __future__ import annotations
from fairchem.core.models.equiformer_v2.global_emb import (
    MOFGlobalPropertyEncoder,
    GlobalConcatFuse,
)
import contextlib
import logging
import time
import typing
from typing import Optional

import torch
import torch.nn as nn

if typing.TYPE_CHECKING:
    from torch_geometric.data.batch import Batch

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin
from fairchem.core.models.escn.escn import eSCN
from fairchem.core.models.escn.so3 import (
    CoefficientMapping,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
)

with contextlib.suppress(ImportError):
    from e3nn import o3

MOF_NAME_LIST_TRAIN = [
    "ABEFUL", "ABESUX", "ABEXEM", "ABEXIQ", "ABEYAJ", "ABEYIR", "ABIXOZ", "ABULOB", "ABUWOJ", "ACAJIY",
    "ACAJIZ", "ACAJOF", "ACIBOE", "ACODON", "ACOGAB", "ACOGEF", "ACOLIP", "ACUBAB", "ACUTOI", "ADABUE",
    "ADAXEK", "ADAXIO", "ADIQEL", "ADOBAX", "ADOBOL", "ADOCEC", "ADODAA", "ADODII", "ADODOO", "ADODUU",
    "ADUWIH", "AFEJOK", "AFENEE", "AFIPAH", "AFITEP", "AFITUF", "AFIXAO", "AFOTOF", "AFOTUL", "AFOVAT",
    "AFOVEX", "AFOVIB", "AFOVOH", "AFOYIE", "AFOYOK", "AGABIU", "AGARUW", "AGONAM", "AGONEQ", "AHAMAY",
    "AHOKIR", "AHOKIR01", "AHOKOX", "AJOTEY", "AJOVEB", "AKEDIF", "AKOBAF", "AKUKUO", "ALAMUW", "ALEQIT",
    "ALIXAU", "ALIXOJ", "ALIXUP", "ALOLIY", "ALUJEX", "ALUKIC", "ALULAV", "ALULID", "AMAFOK", "AMAGAX",
    "AMILUE", "AMIMEP", "AMIWID", "AMIWOJ", "AMIWUP", "AMIXAW", "AMIZIG", "AMOYOR", "AMUFOF", "AMUWUB",
    "AMUXAI", "ANENEN", "ANENIR", "ANOMUM", "ANONAT", "ANUGIA", "ANUGOG", "APACAX", "APARAM", "APAYOH",
    "APAYUN", "APAZIC", "APAZOI", "APEBED", "APEBEE", "AQEQES", "AQEQIW", "AQIQAU", "AQOLEZ", "AQOTIK",
    "AQUCOF", "AQUCOG", "AQUCUM", "AQUDAS", "AQUDAT", "AQUDEX", "AQUDIB", "AQUDOH", "ARAFEF", "ARAFIJ",
    "ARAFOP", "ARAGOR", "ARINOF", "ARINUL", "AROFAP", "AROFET", "ARUYES", "ARUYOB", "ARUYUH", "ARUYUH01",
    "ASACON", "ASADAA", "ASEJOZ", "ASEJOZ01", "ASEKAM", "ASINAT", "ASIQEA", "ASIQIE", "ASOHUL", "ASOSUY",
    "ASOVEL", "ASUWUI", "ATAFIK", "ATAYAX", "ATAYEB", "ATEYED", "ATIBAG", "ATIBOU", "ATIBOU01", "ATIBOU02",
    "ATIBUA", "ATICAH", "ATICEL", "ATIXAE", "ATIXUX", "ATOGEV", "ATOWOW", "ATULIM", "AVAGUB", "AVAQIX",
    "AVAQIX01", "AVEHAM", "AVELOD", "AVELUJ", "AVEMAQ", "AVEQID", "AVESOL", "AVESUR", "AVEVEE", "AVIHIY",
    "AVIHOE", "AVIMIC", "AVIMOI", "AVIMUO", "AVIPAX", "AVIPOL", "AVOSIO", "AWAGEL", "AWAKEQ", "AWEJIW",
    "AWIBAL", "AWOFOI", "AXIMUP", "AXIPEE", "AXIPEE03", "AXUBAW", "AXUBOL", "AZAFIS", "AZEWEH", "AZIVAI",
    "AZIXUD", "BACGAQ", "BAEDTA01", "BAFHAU", "BAKJII", "BAKYOE", "BAMKIM", "BAMKOS", "BAPDAA", "BAQWIC",
    "BARZUR", "BASJUB", "BAXFIR01", "BAXFIR02", "BAXFUD", "BAXGAK", "BAXSIE", "BAZFUF", "BAZFUF01", "BAZJET",
    "BAZJIX", "BEFLUV", "BEFMAC", "BEFPAF", "BEJZAS", "BEKSAM", "BEKYOG", "BELJIL", "BELVEV", "BENXUP",
    "BENZOL", "BENZUR", "BEPBAB", "BEPLUF", "BEPMAM", "BEPMIU", "BEPNIV", "BEPNOB", "BEPNUH", "BEPPAP",
    "BEPPIX", "BEPPOD", "BEPVEZ", "BEPVID", "BEPVOJ", "BEPVOJ01", "BETDAH", "BETDEL", "BETDIP", "BETDUB",
    "BETFAJ", "BETFEN", "BETFUD", "BETGAK", "BETGIS", "BETGOY", "BETHEP", "BEWVOQ", "BEWWEH", "BEXQIF",
    "BEXQOL", "BEXQUR", "BEXRAY", "BEXREC", "BEXRIG", "BEXRUT", "BEXSAA", "BEXSEE", "BEXSII", "BEYKEX",
    "BEYSEF", "BIBBUL", "BIBXOB", "BIBXUH", "BICDAU", "BICPOT", "BIFKEI", "BIMDEF", "BIMDIL", "BIQHEQ",
    "BIQVII", "BIRMUM", "BIRSEC", "BIRSIG", "BIWYAH10", "BIXXIR", "BIYSUZ", "BIYTOU", "BOCKOT", "BOHJOZ",
    "BOJCIN", "BOKQEZ", "BOMCOX", "BOPHEV", "BOPHIZ", "BORBEQ", "BOVHEB", "BOWNUY", "BOWRAH", "BOWZUK",
    "BOXYIY", "BOYTEQ", "BOZFIH", "BUBQOE", "BUCSEY", "BUCWAZ", "BUGSAY", "BUGSEC", "BUGSIG", "BUGSOM",
    "BUKMUQ", "BUKMUQ01", "BUKRUW01", "BUKRUW03", "BUKYAJ", "BUPVEP", "BUQWER", "BURQIQ", "BUSGOM", "BUSNAF",
    "BUSQEM", "BUSQIQ", "BUSRUE", "BUVHEH", "BUVWOF02", "BUVWOF03", "BUVXOG", "BUVYEX", "BUVYIB", "BUVYIB01",
    "BUWMEN", "BUWMIR", "BUYNAL", "CACBUG", "CADPII", "CADQAB", "CADQOP", "CADZIT", "CAGSAG", "CAGXIU",
    "CAHVAJ", "CAJQEL", "CAJQIP", "CAMROA", "CANRIU", "CANRUG", "CAPLOX", "CAVNOE", "CAVRAU", "CAVREY",
    "CAVSUP", "CAVVUS", "CAWDOU02", "CAWDOU", "CAWVIH", "CAXSUR", "CAXVII", "CAXVOO", "CAXVUU", "CAXWAB",
    "CAXWEF", "CAXWIJ", "CAYDEN", "CAYDUD", "CAYMIZ", "CAYQAV", "CAYSIE", "CAYSOK", "CAYYEG", "CEBDOD",
    "CEDFIC", "CEGFAW", "CEGWER", "CEJCAW", "CELZEA", "CENJIQ", "CENKAJ", "CENPUI", "CEQBUV", "CETFOX",
    "CETFUD", "CETGOY", "CETNUL", "CEYPUT", "CEYPUT01", "CIDKUX", "CIFDOM", "CIFDUS", "CIFJAE", "CIFMEL",
    "CIFMIP", "CIGFEF", "CIJVOH", "CIJVOH01", "CILFIN", "CIPCUB", "CISLUN", "CIVTEH", "CIWYUD", "CIXCOC",
    "CIYZUG", "COBWEV", "COCMAJ", "COCMOY", "CODFEH", "CODFUX", "CODMUF", "CODNAM", "CODROD", "CODRUJ",
    "COJGEO", "COJHIT", "COKFAK", "COKMEW", "COKMIA", "COKMOG", "COKMUM", "COKNAT", "COKPEZ", "COKPID",
    "COKPOJ", "COKPUP", "COKQAW", "COKQEA", "COKQOK", "COMDOY", "COQGAS", "COQHUN", "COQNIF", "COQTEJ",
    "COTXEQ", "COTXIU", "COTXOA", "COWBEX", "COWMAD", "COWMEE", "COWMIL", "COWMUX", "COXDIE", "COXHED",
    "COXHIH", "COXHON", "COXXIX", "COYCEZ", "COYCID", "COYTEQ", "COYTIU", "CUBHOY", "CUCKIV", "CUCSAU",
    "CUGVUW", "CUIMDZ01", "CUKXIR", "CUKXOW", "CUKZOY", "CUMDIY", "CUMJEB", "CUMLUT", "CUNFOH", "CUNFOH01",
    "CUNQOT", "CUNQUZ", "CUNWEO", "CUQRIR", "CUQROX", "CUQRUD", "CUQXUI", "CURBOH", "CURQEN", "CURYOE01",
    "CUSDIE", "CUSZEX", "CUSZIB", "CUTGEE", "CUTGII", "CUVGIK", "CUVGOQ", "CUVHIL01", "CUVTUJ", "CUWTUL",
    "CUWVEX", "CUWVIB", "CUWZIF", "CUXFIL", "CUXJEM", "CUXLEO", "CUYWUP", "CUYYAX", "CUYYEB", "CUZBUW",
    "DACFAR", "DACYUE", "DADLIG", "DAFSOV", "DAFYAN", "DAFYOB", "DAGDUL", "DAGJED", "DAJHOM", "DAKVIW",
    "DAKVOC", "DAKVUI", "DALVAQ", "DAMZEZ", "DANZAV", "DANZID", "DANZOJ", "DAPBIH", "DAPYAW", "DARBUU",
    "DARSEV", "DAWBIO", "DAWBOU", "DAWCAH", "DAWCOV", "DAWDAI", "DAWWEF", "DAXHIV", "DAXHIV01", "DAXHUH",
    "DAXHUH01", "DEBGIA", "DEFKUU", "DEFRIR", "DEGJIK", "DEJCIF", "DEJJIN", "DEJRUH", "DEKHEH", "DEKHUX",
    "DEKPUF", "DELGEI", "DEPSEW", "DEPTOH", "DEPXAY", "DEPXOM", "DEQVIF", "DEYDOA", "DEYJIC", "DEYJOI",
    "DEYNIG", "DEZCER", "DEZKAW", "DIBPAH", "DIBPIP", "DICKEH", "DICVIW", "DIDBAV", "DIDDOK", "DIKJUD",
    "DIKPOD", "DINNIZ", "DIPMAS", "DIRPIF", "DITJIB", "DITJOH", "DITTEH", "DITXOU", "DITYEM", "DITYOW",
    "DIVNIH", "DIVPUU", "DIVSUY", "DIWNAA", "DIXKOL", "DIYKAZ", "DIYKED", "DIYTEM", "DOCKOW", "DODMIT",
    "DOGZIJ", "DOHNUL", "DOHYIJ01", "DOKDEN", "DOKHIV", "DOLPIF", "DOLRAZ", "DOLXOR", "DOLXOR01", "DOLXOR02",
    "DONKOI", "DONKUO", "DONNAW", "DONNAW01", "DONNEA", "DONNIE", "DORFOG", "DORYUG", "DOYBEA", "DOYBUQ01",
    "DOYHOQ", "DOYHUW", "DOYLEI", "DOZCEC", "DOZKIO", "DUBKAO", "DUBWON"
]
## 与训练集相互正交
MOF_NAME_LIST_TEST = [
    "ABEFUL", "ABESUX", "ABEXEM", "ABEXIQ", "ABEYAJ", "ABEYIR", "ABIXOZ", "ABULOB", "ABUWOJ", "ACAJIY",
    "ACAJIZ", "ACAJOF", "ACIBOE", "ACODON", "ACOGAB", "ACOGEF", "ACOLIP", "ACUBAB", "ACUTOI", "ADABUE",
    "ADAXEK", "ADAXIO", "ADIQEL", "ADOBAX", "ADOBOL", "ADOCEC", "ADODAA", "ADODII", "ADODOO", "ADODUU",
    "ADUWIH", "AFEJOK", "AFENEE", "AFIPAH", "AFITEP", "AFITUF", "AFIXAO", "AFOTOF", "AFOTUL", "AFOVAT",
    "AFOVEX", "AFOVIB", "AFOVOH", "AFOYIE", "AFOYOK", "AGABIU", "AGARUW", "AGONAM", "AGONEQ", "AHAMAY",
    "AHOKIR", "AHOKIR01", "AHOKOX", "AJOTEY", "AJOVEB", "AKEDIF", "AKOBAF", "AKUKUO", "ALAMUW", "ALEQIT",
    "ALIXAU", "ALIXOJ", "ALIXUP", "ALOLIY", "ALUJEX", "ALUKIC", "ALULAV", "ALULID", "AMAFOK", "AMAGAX",
    "AMILUE", "AMIMEP", "AMIWID", "AMIWOJ", "AMIWUP", "AMIXAW", "AMIZIG", "AMOYOR", "AMUFOF", "AMUWUB",
    "AMUXAI", "ANENEN", "ANENIR", "ANOMUM", "ANONAT", "ANUGIA", "ANUGOG", "APACAX", "APARAM", "APAYOH",
    "APAYUN", "APAZIC", "APAZOI", "APEBED", "APEBEE", "AQEQES", "AQEQIW", "AQIQAU", "AQOLEZ", "AQOTIK",
    "AQUCOF", "AQUCOG", "AQUCUM", "AQUDAS", "AQUDAT", "AQUDEX", "AQUDIB", "AQUDOH", "ARAFEF", "ARAFIJ",
    "ARAFOP", "ARAGOR", "ARINOF", "ARINUL", "AROFAP", "AROFET", "ARUYES", "ARUYOB", "ARUYUH", "ARUYUH01",
    "ASACON", "ASADAA", "ASEJOZ", "ASEJOZ01", "ASEKAM", "ASINAT", "ASIQEA", "ASIQIE", "ASOHUL", "ASOSUY",
    "ASOVEL", "ASUWUI", "ATAFIK", "ATAYAX", "ATAYEB", "ATEYED", "ATIBAG", "ATIBOU", "ATIBOU01", "ATIBOU02",
    "ATIBUA", "ATICAH", "ATICEL", "ATIXAE", "ATIXUX", "ATOGEV", "ATOWOW", "ATULIM", "AVAGUB", "AVAQIX",
    "AVAQIX01", "AVEHAM", "AVELOD", "AVELUJ", "AVEMAQ", "AVEQID", "AVESOL", "AVESUR", "AVEVEE", "AVIHIY",
    "AVIHOE", "AVIMIC", "AVIMOI", "AVIMUO", "AVIPAX", "AVIPOL", "AVOSIO", "AWAGEL", "AWAKEQ", "AWEJIW",
    "AWIBAL", "AWOFOI", "AXIMUP", "AXIPEE", "AXIPEE03", "AXUBAW", "AXUBOL", "AZAFIS", "AZEWEH", "AZIVAI",
    "AZIXUD", "BACGAQ", "BAEDTA01", "BAFHAU", "BAKJII", "BAKYOE", "BAMKIM", "BAMKOS", "BAPDAA", "BAQWIC",
    "BARZUR", "BASJUB", "BAXFIR01", "BAXFIR02", "BAXFUD", "BAXGAK", "BAXSIE", "BAZFUF", "BAZFUF01", "BAZJET",
    "BAZJIX", "BEFLUV", "BEFMAC", "BEFPAF", "BEJZAS", "BEKSAM", "BEKYOG", "BELJIL", "BELVEV", "BENXUP",
    "BENZOL", "BENZUR", "BEPBAB", "BEPLUF", "BEPMAM", "BEPMIU", "BEPNIV", "BEPNOB", "BEPNUH", "BEPPAP",
    "BEPPIX", "BEPPOD", "BEPVEZ", "BEPVID", "BEPVOJ", "BEPVOJ01", "BETDAH", "BETDEL", "BETDIP", "BETDUB",
    "BETFAJ", "BETFEN", "BETFUD", "BETGAK", "BETGIS", "BETGOY", "BETHEP", "BEWVOQ", "BEWWEH", "BEXQIF",
    "BEXQOL", "BEXQUR", "BEXRAY", "BEXREC", "BEXRIG", "BEXRUT", "BEXSAA", "BEXSEE", "BEXSII", "BEYKEX",
    "BEYSEF", "BIBBUL", "BIBXOB", "BIBXUH", "BICDAU", "BICPOT", "BIFKEI", "BIMDEF", "BIMDIL", "BIQHEQ",
    "BIQVII", "BIRMUM", "BIRSEC", "BIRSIG", "BIWYAH10", "BIXXIR", "BIYSUZ", "BIYTOU", "BOCKOT", "BOHJOZ",
    "BOJCIN", "BOKQEZ", "BOMCOX", "BOPHEV", "BOPHIZ", "BORBEQ", "BOVHEB", "BOWNUY", "BOWRAH", "BOWZUK",
    "BOXYIY", "BOYTEQ", "BOZFIH", "BUBQOE", "BUCSEY", "BUCWAZ", "BUGSAY", "BUGSEC", "BUGSIG", "BUGSOM",
    "BUKMUQ", "BUKMUQ01", "BUKRUW01", "BUKRUW03", "BUKYAJ", "BUPVEP", "BUQWER", "BURQIQ", "BUSGOM", "BUSNAF",
    "BUSQEM", "BUSQIQ", "BUSRUE", "BUVHEH", "BUVWOF02", "BUVWOF03", "BUVXOG", "BUVYEX", "BUVYIB", "BUVYIB01",
    "BUWMEN", "BUWMIR", "BUYNAL", "CACBUG", "CADPII", "CADQAB", "CADQOP", "CADZIT", "CAGSAG", "CAGXIU",
    "CAHVAJ", "CAJQEL", "CAJQIP", "CAMROA", "CANRIU", "CANRUG", "CAPLOX", "CAVNOE", "CAVRAU", "CAVREY",
    "CAVSUP", "CAVVUS", "CAWDOU02", "CAWDOU", "CAWVIH", "CAXSUR", "CAXVII", "CAXVOO", "CAXVUU", "CAXWAB",
    "CAXWEF", "CAXWIJ", "CAYDEN", "CAYDUD", "CAYMIZ", "CAYQAV", "CAYSIE", "CAYSOK", "CAYYEG", "CEBDOD",
    "CEDFIC", "CEGFAW", "CEGWER", "CEJCAW", "CELZEA", "CENJIQ", "CENKAJ", "CENPUI", "CEQBUV", "CETFOX",
    "CETFUD", "CETGOY", "CETNUL", "CEYPUT", "CEYPUT01", "CIDKUX", "CIFDOM", "CIFDUS", "CIFJAE", "CIFMEL",
    "CIFMIP", "CIGFEF", "CIJVOH", "CIJVOH01", "CILFIN", "CIPCUB", "CISLUN", "CIVTEH", "CIWYUD", "CIXCOC",
    "CIYZUG", "COBWEV", "COCMAJ", "COCMOY", "CODFEH", "CODFUX", "CODMUF", "CODNAM", "CODROD", "CODRUJ",
    "COJGEO", "COJHIT", "COKFAK", "COKMEW", "COKMIA", "COKMOG", "COKMUM", "COKNAT", "COKPEZ", "COKPID",
    "COKPOJ", "COKPUP", "COKQAW", "COKQEA", "COKQOK", "COMDOY", "COQGAS", "COQHUN", "COQNIF", "COQTEJ",
    "COTXEQ", "COTXIU", "COTXOA", "COWBEX", "COWMAD", "COWMEE", "COWMIL", "COWMUX", "COXDIE", "COXHED",
    "COXHIH", "COXHON", "COXXIX", "COYCEZ", "COYCID", "COYTEQ", "COYTIU", "CUBHOY", "CUCKIV", "CUCSAU",
    "CUGVUW", "CUIMDZ01", "CUKXIR", "CUKXOW", "CUKZOY", "CUMDIY", "CUMJEB", "CUMLUT", "CUNFOH", "CUNFOH01",
    "CUNQOT", "CUNQUZ", "CUNWEO", "CUQRIR", "CUQROX", "CUQRUD", "CUQXUI", "CURBOH", "CURQEN", "CURYOE01",
    "CUSDIE", "CUSZEX", "CUSZIB", "CUTGEE", "CUTGII", "CUVGIK", "CUVGOQ", "CUVHIL01", "CUVTUJ", "CUWTUL",
    "CUWVEX", "CUWVIB", "CUWZIF", "CUXFIL", "CUXJEM", "CUXLEO", "CUYWUP", "CUYYAX", "CUYYEB", "CUZBUW",
    "DACFAR", "DACYUE", "DADLIG", "DAFSOV", "DAFYAN", "DAFYOB", "DAGDUL", "DAGJED", "DAJHOM", "DAKVIW",
    "DAKVOC", "DAKVUI", "DALVAQ", "DAMZEZ", "DANZAV", "DANZID", "DANZOJ", "DAPBIH", "DAPYAW", "DARBUU",
    "DARSEV", "DAWBIO", "DAWBOU", "DAWCAH", "DAWCOV", "DAWDAI", "DAWWEF", "DAXHIV", "DAXHIV01", "DAXHUH",
    "DAXHUH01", "DEBGIA", "DEFKUU", "DEFRIR", "DEGJIK", "DEJCIF", "DEJJIN", "DEJRUH", "DEKHEH", "DEKHUX",
    "DEKPUF", "DELGEI", "DEPSEW", "DEPTOH", "DEPXAY", "DEPXOM", "DEQVIF", "DEYDOA", "DEYJIC", "DEYJOI",
    "DEYNIG", "DEZCER", "DEZKAW", "DIBPAH", "DIBPIP", "DICKEH", "DICVIW", "DIDBAV", "DIDDOK", "DIKJUD",
    "DIKPOD", "DINNIZ", "DIPMAS", "DIRPIF", "DITJIB", "DITJOH", "DITTEH", "DITXOU", "DITYEM", "DITYOW",
    "DIVNIH", "DIVPUU", "DIVSUY", "DIWNAA", "DIXKOL", "DIYKAZ", "DIYKED", "DIYTEM", "DOCKOW", "DODMIT",
    "DOGZIJ", "DOHNUL", "DOHYIJ01", "DOKDEN", "DOKHIV", "DOLPIF", "DOLRAZ", "DOLXOR", "DOLXOR01", "DOLXOR02",
    "DONKOI", "DONKUO", "DONNAW", "DONNAW01", "DONNEA", "DONNIE", "DORFOG", "DORYUG", "DOYBEA", "DOYBUQ01",
    "DOYHOQ", "DOYHUW", "DOYLEI", "DOZCEC", "DOZKIO", "DUBKAO", "DUBWON"
]
# ============================================================================
# LoRALinear类（从equiformer_lora.py复制）
# ============================================================================

class LoRALinear(nn.Module):
    """
    通用的LoRA线性层实现，适用于任何nn.Linear层
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_scale: float = 0.01,
        enable_lora: bool = True,
    ):
        super().__init__()
        assert isinstance(original_layer, nn.Linear), "original_layer 必须是 nn.Linear"

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.enable_lora = enable_lora
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # 冻结原始权重
        for p in self.original_layer.parameters():
            p.requires_grad = False

        weight_dtype = self.original_layer.weight.dtype
        weight_device = self.original_layer.weight.device

        if enable_lora and rank > 0:
            self.lora_A = nn.Parameter(
                torch.randn(rank, self.in_features, device=weight_device, dtype=weight_dtype) * init_scale
            )
            self.lora_B = nn.Parameter(
                torch.zeros(self.out_features, rank, device=weight_device, dtype=weight_dtype)
            )
        else:
            self.lora_A = None
            self.lora_B = None

        self.merged = False

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or (self.lora_A is None) or (not self.enable_lora):
            return self.original_layer(x)

        result = self.original_layer(x)

        # 如果lora_B全为0（初始化状态），直接返回
        if self.lora_B is not None and torch.norm(self.lora_B) < 1e-6:
            return result

        lora_intermediate = self.dropout(x) @ self.lora_A.t()
        lora_update = lora_intermediate @ self.lora_B.t()
        result = result + lora_update * self.scaling
        return result

    @torch.no_grad()
    def merge(self):
        """将LoRA权重合并到原始权重"""
        if self.merged or self.lora_A is None:
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.original_layer.weight.data += delta_w.to(self.original_layer.weight.dtype)
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """取消合并"""
        if not self.merged or self.lora_A is None:
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.original_layer.weight.data -= delta_w.to(self.original_layer.weight.dtype)
        self.merged = False


# ============================================================================
# eSCN LoRA模型
# ============================================================================

@registry.register_model("escn_lora")
class eSCNLoRA(eSCN):
    """
    eSCN模型 + LoRA微调版本 + 全局特征支持
    """
    
    def __init__(
        self,
        # eSCN的所有原有参数
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        regress_forces: bool = True,
        otf_graph: bool = False,
        max_neighbors: int = 40,
        cutoff: float = 8.0,
        max_num_elements: int = 90,
        num_layers: int = 8,
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        sphere_channels: int = 128,
        hidden_channels: int = 256,
        edge_channels: int = 128,
        num_sphere_samples: int = 128,
        distance_function: str = "gaussian",
        basis_width_scalar: float = 1.0,
        distance_resolution: float = 0.02,
        show_timing_info: bool = False,
        resolution: int | None = None,
        activation_checkpoint: bool | None = False,
        
        # LoRA特定参数
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        
        # 全局特征参数
        use_mof_global_features: bool = True,
        mof_global_dim: int = 32,
        global_fusion_type: str = "additive",
        mof_encoder_excel_path: Optional[str] = None,
        
        # 预训练模型路径
        pretrained_model_path: Optional[str] = None,
    ):
        # 先初始化eSCN基类
        super().__init__(
            use_pbc=use_pbc,
            use_pbc_single=use_pbc_single,
            regress_forces=regress_forces,
            otf_graph=otf_graph,
            max_neighbors=max_neighbors,
            cutoff=cutoff,
            max_num_elements=max_num_elements,
            num_layers=num_layers,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            edge_channels=edge_channels,
            num_sphere_samples=num_sphere_samples,
            distance_function=distance_function,
            basis_width_scalar=basis_width_scalar,
            distance_resolution=distance_resolution,
            show_timing_info=show_timing_info,
            resolution=resolution,
            activation_checkpoint=activation_checkpoint,
        )
        
        # LoRA配置
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # LoRA目标模块（eSCN特定的模块路径）
        if lora_target_modules is None:
            # eSCN的默认目标模块
            lora_target_modules = [
                'layer_blocks.fc1_sphere',     # LayerBlock中的fc层
                'layer_blocks.fc2_sphere',
                'layer_blocks.fc3_sphere',
                'layer_blocks.message_block.fc1_dist',   # MessageBlock中的fc层
                'layer_blocks.message_block.fc1_r',
                'layer_blocks.message_block.fc2_r',
                'layer_blocks.message_block.fc1_i',
                'layer_blocks.message_block.fc2_i',
                'energy_block.fc1',            # EnergyBlock中的fc层
                'energy_block.fc2',
                'force_block.fc1',              # ForceBlock中的fc层（如果存在）
                'force_block.fc2',
            ]
        self.lora_target_modules = lora_target_modules
        
        self.lora_config = {
            'rank': self.lora_rank,
            'alpha': self.lora_alpha,
            'dropout': self.lora_dropout,
            'target_modules': self.lora_target_modules,
        }
        self.lora_modules = {}
        
        # ====================================================================
        # 全局特征模块初始化
        # ====================================================================
        self.use_mof_global_features = use_mof_global_features
        self.mof_global_dim = mof_global_dim
        self.global_fusion_type = global_fusion_type
        
        if self.use_mof_global_features:
            # 默认 excel 路径（如果没有提供）
            if mof_encoder_excel_path is None:
                mof_encoder_excel_path = '/home/dell/autodl-tmp/lorafair/data/MOF_embedding_train-check580.xlsx'
            
            self.mof_global_encoder = MOFGlobalPropertyEncoder(
                excel_path=mof_encoder_excel_path,
                global_dim=mof_global_dim,
                multi_label_pool='mean',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.global_fusion = GlobalConcatFuse(
                node_channels=self.sphere_channels,
                proj_dim=mof_global_dim,
            )
            
            # 定义 MOF 名称列表（可以从配置文件读取）
            # 这里使用简化的列表，实际使用时建议从 equiformer_lora.py 导入
            self.MOF_NAME_LIST_TRAIN = None  # 可以后续从配置文件加载
            self.MOF_NAME_LIST_TEST = None
        else:
            self.mof_global_encoder = None
            self.global_fusion = None
        
        # 加载预训练权重（如果有）
        if pretrained_model_path is not None:
            self._load_pretrained_weights(pretrained_model_path)
            logging.info("预训练模型加载完成")
        else:
            # 如果没加载预训练权重，使用原始初始化
            pass
        
        # 应用LoRA
        print(f"use_lora: {self.use_lora}")
        if self.use_lora:
            self._apply_lora()
        
        # 冻结基础模型参数
        self._freeze_base_model()
    
    # ... 保持原有的 _load_pretrained_weights, _apply_lora 等方法不变 ...
    
    def _freeze_base_model(self):
        """冻结基础模型参数，只保留LoRA参数和全局特征模块可训练"""
        frozen_params = 0
        trainable_params = 0
        lora_params = 0
        encoder_params = 0
        
        for name, param in self.named_parameters():
            # LoRA参数始终可训练
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
                lora_params += param.numel()
                trainable_params += param.numel()
                continue
            
            # 全局特征编码器参数可训练
            if self.use_mof_global_features and 'mof_global_encoder' in name:
                param.requires_grad = True
                encoder_params += param.numel()
                trainable_params += param.numel()
                continue
            
            # 全局融合模块参数可训练
            if self.use_mof_global_features and 'global_fusion' in name:
                param.requires_grad = True
                encoder_params += param.numel()
                trainable_params += param.numel()
                continue
            
            # 其他参数都冻结
            param.requires_grad = False
            frozen_params += param.numel()
        
        total_params = frozen_params + trainable_params
        
        print("=" * 50)
        print("eSCN LoRA Parameter Summary:")
        print(f"Frozen parameters:     {frozen_params:,}")
        print(f"LoRA parameters:       {lora_params:,}")
        if self.use_mof_global_features:
            print(f"Global encoder params:  {encoder_params:,}")
        print(f"Total trainable:       {trainable_params:,}")
        print(f"Total parameters:      {total_params:,}")
        print(f"Trainable ratio:       {trainable_params/total_params:.2%}")
        print("=" * 50)
    
    # ====================================================================
    # 覆盖 forward 方法以集成全局特征
    # ====================================================================
    
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        device = data.pos.device
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype

        start_time = time.time()
        atomic_numbers = data.atomic_numbers.long()
        assert (
            atomic_numbers.max().item() < self.max_num_elements
        ), "Atomic number exceeds that given in model config"
        num_atoms = len(atomic_numbers)
        
        # 🎯 获取 MOF 全局特征
        mof_global_embedding = None
        if self.use_mof_global_features and self.mof_global_encoder is not None:
            mof_names_batch = []
            if hasattr(data, 'name') and data.name is not None:
                # data.name 通常是list，长度为batch_size
                if isinstance(data.name, list):
                    for i in range(min(len(data.name), self.batch_size)):
                        name = str(data.name[i])
                        # 提取MOF名称（假设格式为 "MOFNAME_defect"）
                        mof_name = name.split('_')[0] if '_' in name else name
                        mof_names_batch.append(mof_name)
                    # 如果name列表长度不够，用最后一个补齐
                    while len(mof_names_batch) < self.batch_size:
                        mof_names_batch.append(mof_names_batch[-1] if len(mof_names_batch) > 0 else "")
                else:
                    # 如果不是list，尝试用第一个（fallback）
                    name = str(data.name[0]) if hasattr(data.name, '__getitem__') else str(data.name)
                    mof_name = name.split('_')[0] if '_' in name else name
                    mof_names_batch = [mof_name] * self.batch_size
            
            # 为每个样本编码MOF全局特征
            if len(mof_names_batch) > 0:
                mof_embeddings_list = []
                for mof_name in mof_names_batch:
                    if mof_name:
                        emb = self.mof_global_encoder(mof_name)  # [1, global_dim]
                        mof_embeddings_list.append(emb)
                    else:
                        # 如果MOF名称无效，使用零向量
                        emb = torch.zeros(
                            1, 
                            self.mof_global_dim, 
                            device=device, 
                            dtype=self.dtype
                        )
                        mof_embeddings_list.append(emb)
                
                # 拼接成 [batch_size, global_dim]
                mof_global_embedding = torch.cat(mof_embeddings_list, dim=0)
        
        graph = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        self.SO3_edge_rot = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_edge_rot.append(SO3_Rotation(edge_rot_mat, self.lmax_list[i]))

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l=0,m=0 coefficients for each resolution
        for i in range(self.num_resolutions):
            x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                :, offset : offset + self.sphere_channels
            ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # This can be expensive to compute (not implemented efficiently), so only do it once and pass it along to each layer
        mappingReduced = CoefficientMapping(self.lmax_list, self.mmax_list, device)

        ###############################################################
        # 🎯 集成全局特征（在进入 layer_blocks 之前）
        ###############################################################
        if self.use_mof_global_features and mof_global_embedding is not None and self.global_fusion is not None:
            # 提取 l=0, m=0 的节点特征（标量特征，适合融合全局信息）
            l0_m0_features = x.embedding[:, 0, :]  # [num_atoms, sphere_channels]
            # 使用全局融合模块
            enhanced_l0_m0 = self.global_fusion(
                l0_m0_features, 
                mof_global_embedding, 
                data.batch[:num_atoms] if hasattr(data, 'batch') else torch.zeros(num_atoms, dtype=torch.long, device=device)
            )
            # 更新嵌入
            x.embedding[:, 0, :] = enhanced_l0_m0

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            if self.activation_checkpoint:
                x_message = torch.utils.checkpoint.checkpoint(
                    self.layer_blocks[i],
                    x,
                    atomic_numbers,
                    graph.edge_distance,
                    graph.edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                    use_reentrant=not self.training,
                )
            else:
                x_message = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    graph.edge_distance,
                    graph.edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                )

            if i > 0:
                # Residual layer for all layers past the first
                x.embedding = x.embedding + x_message.embedding
            else:
                x = x_message

        # Sample the spherical channels (node embeddings) at evenly distributed points on the sphere.
        # These values are fed into the output blocks.
        x_pt = torch.tensor([], device=device)
        offset = 0
        # Compute the embedding values at every sampled point on the sphere
        for i in range(self.num_resolutions):
            num_coefficients = int((x.lmax_list[i] + 1) ** 2)
            x_pt = torch.cat(
                [
                    x_pt,
                    torch.einsum(
                        "abc, pb->apc",
                        x.embedding[:, offset : offset + num_coefficients],
                        self.sphharm_weights[i],
                    ).contiguous(),
                ],
                dim=2,
            )
            offset = offset + num_coefficients

        x_pt = x_pt.view(-1, self.sphere_channels_all)

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x_pt)
        energy = torch.zeros(len(data.natoms), device=device)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        # Scale energy to help balance numerical precision w.r.t. forces
        energy = energy * 0.001

        outputs = {"energy": energy}
        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            forces = self.force_block(x_pt, self.sphere_points)
            outputs["forces"] = forces

        if self.show_timing_info is True:
            torch.cuda.synchronize()
            logging.info(
                f"{self.counter} Time: {time.time() - start_time}\tMemory: {len(data.pos)}\t{torch.cuda.max_memory_allocated() / 1000000}"
            )

        self.counter = self.counter + 1

        return outputs
    
    def get_lora_parameters(self):
        """获取所有LoRA参数"""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_params.append(param)
        return lora_params