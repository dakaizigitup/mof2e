
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import re
import yaml
import loralib as lora  # LoRA微调工具包
from fairchem.core.common import gp_utils

# 注册表系统，用来自动管理模型（比如用字符串就能加载）
from fairchem.core.common.registry import registry
# 导入EquiformerV2主体结构和SO3相关模块
from fairchem.core.models.equiformer_v2.equiformer_v2_deprecated import EquiformerV2
from fairchem.core.models.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
# 自定义的全局嵌入模块
from model_lora.global_emb import MOFGlobalPropertyEncoder,EquivariantGlobalFusion

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



# ===============================================
# ============== LoRA 核心模块 ==================
# ===============================================
class SO3LoRAWrapper(nn.Module):
    """专门为SO3_LinearV2设计的LoRA包装器"""
    def __init__(
        self,
        original_layer,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer   # 原始线性层
        
        # 冻结原始参数
        for param in self.original_layer.parameters():
            param.requires_grad = False
        # 检查是否是 SO3 等变层（Equiformer 使用的特殊线性层）
        # 为SO3_LinearV2创建等变LoRA
        if hasattr(original_layer, 'irreps_in') and hasattr(original_layer, 'irreps_out'):
            # 使用e3nn确保等变性
            from e3nn import o3
            self.lora = o3.Linear(
                irreps_in=original_layer.irreps_in,
                irreps_out=original_layer.irreps_out,
                bias=False
            )
            # 缩放LoRA输出
            self.scaling = alpha / rank
        elif isinstance(original_layer, nn.Linear) :
            # 标准线性层
            self.lora = lora.Linear(
                original_layer.in_features,
                original_layer.out_features,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout
            )
        elif isinstance(original_layer, nn.Embedding):
            # 如果是embedding层
            self.lora = lora.Embedding(
                original_layer.num_embeddings,
                original_layer.embedding_dim,
                r=rank,
                lora_alpha=alpha,
                merge_weights=True
            )

        else:
            self.lora = None
            
    def forward(self, x):
        # 原始层的输出
        original_out = self.original_layer(x)
        # 如果加了LoRA层，就叠加上去
        if self.lora is not None:
            if not hasattr(self.lora, 'scaling'):
                # SO3线性层
                # SO3层的LoRA输出需要缩放
                lora_out = self.lora(x) * self.scaling
            else:
                # 标准LoRA,embedding
                lora_out = self.lora(x)
            return original_out + lora_out
        return original_out

# ===============================================
# ============ 通用LoRA线性层实现 ================
# ===============================================
class LoRALinear(nn.Module):
    """
    针对普通 nn.Linear 的 LoRA 实现
    原理：冻结原权重，只学习两个小矩阵A、B
    公式：输出 = Wx + BAx * scaling
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # 冻结原始权重
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # 定义LoRA低秩分解矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scaling = alpha / rank
    
    def forward(self, x):
        # 原始输出
        original_output = self.original_layer(x)
        
        # LoRA输出
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        # 最终输出 = 原始输出 + LoRA修正
        return original_output + lora_output * self.scaling


        




@registry.register_model("equiformer_v2_lora")
class EquiformerV2LoRA(EquiformerV2):
    """
    这是整个代码的核心模型：
    - 基于 EquiformerV2（等变图Transformer）
    - 加上 LoRA 微调机制
    - 并融合 MOF 材料的全局特征
    """
    
    def __init__(
        self,
        # === Equiformer 参数 ===
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        regress_forces: bool = True,
        otf_graph: bool = True,
        max_neighbors: int = 500,
        max_radius: float = 5.0,
        max_num_elements: int = 90,
        num_layers: int = 12,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 128,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,
        norm_type: str = "rms_norm_sh",
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        grid_resolution: int | None = None,
        num_sphere_samples: int = 128,
        edge_channels: int = 128,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        use_m_share_rad: bool = False,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        attn_activation: str = "scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
        weight_init: str = "normal",
        enforce_max_neighbors_strictly: bool = True,
        avg_num_nodes: float | None = None,
        avg_degree: float | None = None,
        use_energy_lin_ref: bool | None = False,
        load_energy_lin_ref: bool | None = False,
        
        # === LoRA 参数 ===
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 64.0,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        # 全局嵌入有关
        use_mof_global_features: bool = True,
        mof_global_dim: int = 128,

        global_fusion_type: str = "additive",
        ## pretraine path
        pretrained_model_path: Optional[str] = None,
    ):
        # 调用父类初始化
        super().__init__()
        # 初始化全局嵌入模块
        self.use_mof_global_features = use_mof_global_features
        self.mof_global_encoder = MOFGlobalPropertyEncoder(
                excel_path='/home/dell/autodl-tmp/lorafair/data/MOF_embedding_train-check580.xlsx',
                global_dim=128,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        # 融合模块：把 MOF 的全局特征融合进节点特征
        self.global_fusion = EquivariantGlobalFusion(
                sphere_channels=self.sphere_channels,
                global_dim=mof_global_dim,
                fusion_type=global_fusion_type,
            )

        
        # LoRA配置
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # 要应用LoRA的模块名
        if lora_target_modules is None:
            lora_target_modules = [
                'ga.fc_m0',      # 针对 ga 模块中的 fc_m0
                'ffn.scalar_mlp' # 针对 ffn 模块中的 scalar_mlp
            ]
        self.lora_target_modules = lora_target_modules
        
        
        # 如果启用LoRA，则在指定模块上应用
        if self.use_lora:
            self._apply_lora()
            
        # 冻结基础模型参数
        self._freeze_base_model()
    # ===== 应用LoRA到Transformer层 =====
    def _apply_lora(self):
        print("Applying LoRA to model...")
        
        # 2. 对Transformer块应用LoRA
        for i, block in enumerate(self.blocks):
            # 应用到 ga.fc_m0
            if hasattr(block, 'ga') and hasattr(block.ga, 'fc_m0'):
                self._apply_lora_to_block(block.ga.fc_m0, f"blocks.{i}.ga.fc_m0")
            
            # 应用到 ffn.scalar_mlp
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'scalar_mlp'):
                self._apply_lora_to_block(block.ffn.scalar_mlp, f"blocks.{i}.ffn.scalar_mlp")
        
        print(f"Applied LoRA to {len(self.lora_modules)} modules")
        
    
    def _apply_lora_to_block(self, block, block_name: str):
        """对指定模块（例如 fc_m0）逐层替换成 LoRA 版本"""
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{block_name}.{name}"
                if self._should_apply_lora(full_name):
                    # 替换成LoRA版本
                    parent_module = block
                    module_path = name.split('.')
                    
                    # 导航到直接父模块
                    for path_part in module_path[:-1]:
                        parent_module = getattr(parent_module, path_part)
                    
                    # 替换最后一层
                    child_name = module_path[-1]
                    self._replace_with_lora(parent_module, child_name, module)
    
    def _replace_with_lora(self, parent_module: nn.Module, child_name: str, child_module: nn.Linear):
        """
        将线性层替换为LoRA层
        
        Args:
            parent_module: 父模块
            child_name: 子模块名称
            child_module: 要替换的线性层
        """
        # 创建LoRA层
        lora_layer = LoRALinear(
            original_layer=child_module,
            rank=self.lora_config['rank'],
            alpha=self.lora_config['alpha'],
            dropout=self.lora_config['dropout']
        )
        
        # 替换模块
        setattr(parent_module, child_name, lora_layer)
        # 记录替换的模块
        full_name = f"{parent_module.__class__.__name__}.{child_name}"
        self.lora_modules[full_name] = lora_layer
        print(f"Applied LoRA to: {full_name}")
    
    def _should_apply_lora(self, full_name: str) -> bool:
        """判断是否应该对该线性层应用LoRA"""
        for target_module in self.lora_config['target_modules']:
            if target_module in full_name:
                return True
        return False
    
    # def _replace_with_lora(self, parent_module, name: str, original_module):
    #     """替换模块为LoRA版本"""
    #     module_names = name.split('.')
    #     current_module = parent_module
        
    #     # 导航到父模块
    #     for module_name in module_names[:-1]:
    #         current_module = getattr(current_module, module_name)
        
    #     # 替换最后一层
    #     final_name = module_names[-1]
    #     lora_wrapper = SO3LoRAWrapper(
    #         original_module,
    #         rank=self.lora_rank,
    #         alpha=self.lora_alpha,
    #         dropout=self.lora_dropout
    #     )
    #     setattr(current_module, final_name, lora_wrapper)
    # ===== 冻结除LoRA外的参数 =====
    def _freeze_base_model(self):
        """
        冻结基础模型参数，但保持指定模块可训练
        
        可训练的参数包括：
        1. LoRA参数 (lora_A, lora_B)
        2. self.trainable_modules 中指定的模块
        """
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
            
            # 检查是否属于需要保持可训练的模块
            should_train = any(module_name in name for module_name in self.trainable_modules)
            
            if should_train:
                param.requires_grad = True
                if 'mof_global_encoder' in name:
                    encoder_params += param.numel()
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        total_params = frozen_params + trainable_params
        
        print("=" * 50)
        print("Parameter Freezing Summary:")
        print(f"Frozen parameters:           {frozen_params:,}")
        print(f"LoRA parameters:            {lora_params:,}")
        print(f"MOF encoder parameters:     {encoder_params:,}")
        print(f"Total trainable parameters: {trainable_params:,}")
        print(f"Total parameters:           {total_params:,}")
        print(f"Trainable ratio:            {trainable_params/total_params:.2%}")
        print("=" * 50)
    
    def forward(self, data):
        """
        这是模型前向传播流程的主要逻辑：
        1️⃣ 提取名称 → 获取全局特征
        2️⃣ 构造原子图结构（节点、边、距离等）
        3️⃣ 节点嵌入 + 边特征嵌入
        4️⃣ 融合 MOF 全局特征
        5️⃣ 通过 Transformer 块
        6️⃣ 输出能量 + 力
        """
        # 根据data.name查找对应的待添加的embedding
        mof_name = data.name.split('_')[0]
        if self.training:
            mof_list = MOF_NAME_LIST_TRAIN
        else:
            mof_list = MOF_NAME_LIST_TEST

        if mof_name not in mof_list:
            pass
        else:
            """重写前向传播以集成自定义embedding"""
            defect = data.name.split('_')[1]

            mof_global_embedding = self.mof_global_encoder(mof_name, defect)
            # 调用父类的大部分前向传播逻辑
            self.batch_size = len(data.natoms)
            self.dtype = data.pos.dtype
            self.device = data.pos.device
            atomic_numbers = data.atomic_numbers.long()
            
            # === 生成图结构（节点-边） ===
            graph = self.generate_graph(
                data,
                enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            )
            data_batch = data.batch
            if gp_utils.initialized():
                (
                    atomic_numbers,
                    data_batch,
                    node_offset,
                    edge_index,
                    edge_distance,
                    edge_distance_vec,
                ) = self._init_gp_partitions(
                    graph.atomic_numbers_full,
                    graph.batch_full,
                    graph.edge_index,
                    graph.edge_distance,
                    graph.edge_distance_vec,
                )
                graph.node_offset = node_offset
                graph.edge_index = edge_index
                graph.edge_distance = edge_distance
                graph.edge_distance_vec = edge_distance_vec
            # ... 图并行处理和边旋转矩阵初始化（与原始相同）...
            
            # 初始化边旋转矩阵
            edge_rot_mat = self._init_edge_rot_mat(
                data, graph.edge_index, graph.edge_distance_vec
            )
            
            for i in range(self.num_resolutions):
                self.SO3_rotation[i].set_wigner(edge_rot_mat)
            
            # 🎯 关键修改：集成自定义embedding
            x = SO3_Embedding(
                len(atomic_numbers),
                self.lmax_list,
                self.sphere_channels,
                self.device,
                self.dtype,
            )
            
            offset_res = 0
            offset = 0
            for i in range(self.num_resolutions):
                if self.num_resolutions == 1:
                    x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)#将原子的嵌入赋值到对应的节点位置 130,128
                else:
                    x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                        :, offset : offset + self.sphere_channels
                    ]
                offset = offset + self.sphere_channels
                offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)
            
            # 继续原始的前向传播流程
            graph.edge_distance = self.distance_expansion(graph.edge_distance)
            
            if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
                source_element = graph.atomic_numbers_full[graph.edge_index[0]]
                target_element = graph.atomic_numbers_full[graph.edge_index[1]]
                source_embedding = self.source_embedding(source_element)
                target_embedding = self.target_embedding(target_element)
                graph.edge_distance = torch.cat(
                    (graph.edge_distance, source_embedding, target_embedding), dim=1
                )
            
            # 边度数embedding
            edge_degree = self.edge_degree_embedding(
                graph.atomic_numbers_full,
                graph.edge_distance,
                graph.edge_index,
                len(atomic_numbers),
                getattr(graph, 'node_offset', 0),
            )
            x.embedding = x.embedding + edge_degree.embedding
            # 全局向量的嵌入
            if self.use_mof_global_features and mof_global_embedding is not None:
                l0_m0_features = x.embedding[:, 0, :]  # [num_nodes, sphere_channels]
                enhanced_l0_m0 = self.global_fusion(
                    l0_m0_features, 
                    mof_global_embedding, 
                    data.batch[:len(atomic_numbers)]
                )
                x.embedding[:, 0, :] = enhanced_l0_m0
            # Transformer块处理
            for i in range(self.num_layers):
                x = self.blocks[i](
                    x,
                    graph.atomic_numbers_full,
                    graph.edge_distance,
                    graph.edge_index,
                    batch=getattr(data, 'batch', None),
                    node_offset=getattr(graph, 'node_offset', 0),
                )
            
            # 最终归一化和输出
            x.embedding = self.norm(x.embedding)
            
            # 能量预测
            node_energy = self.energy_block(x)
            node_energy = node_energy.embedding.narrow(1, 0, 1)
            
            energy = torch.zeros(
                len(data.natoms),
                device=node_energy.device,
                dtype=node_energy.dtype,
            )
            energy.index_add_(0, getattr(graph, 'batch_full', data.batch), node_energy.view(-1))
            energy = energy / self.avg_num_nodes
            
            outputs = {"energy": energy}
            
            # 力预测
            if self.regress_forces:
                forces = self.force_block(
                    x,
                    graph.atomic_numbers_full,
                    graph.edge_distance,
                    graph.edge_index,
                    node_offset=getattr(graph, 'node_offset', 0),
                )
                forces = forces.embedding.narrow(1, 1, 3)  # 提取向量分量
                forces = forces.view(-1, 3).contiguous()
                outputs["forces"] = forces
            
            return outputs
    
    def get_lora_parameters(self):
        """获取所有LoRA参数"""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_params.append(param)
        return lora_params
    
    
    

    

# 等变性验证工具
class EquivarianceValidator:
    """
    用来测试模型是否满足物理等变性：
    - 能量应对旋转/平移不变
    - 力应对旋转等变
    """
    
    @staticmethod
    def test_rotation_equivariance(model, data, tolerance=1e-5):
        """测试旋转等变性"""
        from scipy.spatial.transform import Rotation
        
        model.eval()
        with torch.no_grad():
            # 原始预测
            output_original = model(data)
            
            # 随机旋转
            R = torch.tensor(
                Rotation.random().as_matrix(), 
                dtype=torch.float32, 
                device=data.pos.device
            )
            
            # 旋转数据
            data_rotated = data.clone()
            data_rotated.pos = data.pos @ R.T
            
            # 旋转后预测
            output_rotated = model(data_rotated)
            
            # 检查能量不变性
            energy_error = torch.abs(
                output_original["energy"] - output_rotated["energy"]
            ).max().item()
            
            results = {
                "energy_invariant": energy_error < tolerance,
                "energy_error": energy_error,
            }
            
            # 检查力的等变性
            if "forces" in output_original:
                forces_original = output_original["forces"]
                forces_rotated = output_rotated["forces"]
                forces_expected = forces_original @ R.T
                
                force_error = torch.norm(
                    forces_rotated - forces_expected
                ).item()
                
                results.update({
                    "forces_equivariant": force_error < tolerance,
                    "forces_error": force_error,
                })
            
            return results
    
    @staticmethod
    def test_translation_invariance(model, data, tolerance=1e-5):
        """测试平移不变性"""
        model.eval()
        with torch.no_grad():
            # 原始预测
            output_original = model(data)
            
            # 随机平移
            translation = torch.randn(3, device=data.pos.device) * 5.0
            
            # 平移数据
            data_translated = data.clone()
            data_translated.pos = data.pos + translation
            
            # 平移后预测
            output_translated = model(data_translated)
            
            # 检查不变性
            energy_error = torch.abs(
                output_original["energy"] - output_translated["energy"]
            ).max().item()
            
            results = {
                "energy_invariant": energy_error < tolerance,
                "energy_error": energy_error,
            }
            
            if "forces" in output_original:
                force_error = torch.norm(
                    output_original["forces"] - output_translated["forces"]
                ).item()
                
                results.update({
                    "forces_invariant": force_error < tolerance,
                    "forces_error": force_error,
                })
            
            return results

# 训练工具
class LoRATrainer:
    """
    用于管理训练的辅助类：
    - 优化器、调度器
    - 单步训练
    - 等变性验证
    """
    
    def __init__(
        self,
        model: EquiformerV2LoRA,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # 设置优化器
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # 验证器
        self.validator = EquivarianceValidator()
    
    def _setup_optimizer(self):
        """设置优化器"""
        # 分组参数：LoRA参数和自定义embedding参数可能需要不同学习率
        param_groups = []
        
        lora_params = self.model.get_lora_parameters()
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
                'name': 'lora'
            })
        
        
        if not param_groups:
            raise ValueError("No trainable parameters found!")
        
        return torch.optim.AdamW(param_groups)
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(self.optimizer, T_max=1000)
    
    def train_step(self, batch, loss_fn):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        output = self.model(batch)
        
        # 计算损失
        loss = loss_fn(output, batch)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_parameters(), 
            max_norm=1.0
        )
        
        # 优化器步骤
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def validate_equivariance(self, data_sample):
        """验证等变性"""
        rotation_results = self.validator.test_rotation_equivariance(
            self.model, data_sample
        )
        translation_results = self.validator.test_translation_invariance(
            self.model, data_sample
        )
        
        return {
            "rotation": rotation_results,
            "translation": translation_results,
        }

# 使用示例
def create_lora_model_example(config_path: str = 'config.yaml'):
    """创建LoRA模型示例"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    
    # 创建模型
    model = EquiformerV2LoRA(**config)
    
    # 打印模型信息
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return model

def train_lora_model_example():
    """LoRA训练示例"""
    
    # 创建模型
    model = create_lora_model_example()
    
    # 创建训练器
    trainer = LoRATrainer(
        model=model,
        learning_rate=1e-4,
        weight_decay=1e-5,
        warmup_steps=500,
    )
    
    # 损失函数
    def loss_fn(output, batch):
        energy_loss = F.mse_loss(output["energy"], batch.energy)
        total_loss = energy_loss
        
        if "forces" in output and hasattr(batch, 'forces'):
            force_loss = F.mse_loss(output["forces"], batch.forces)
            total_loss += force_loss
        
        return total_loss
    
    # 训练循环示例
    print("Starting LoRA fine-tuning...")
    
    # 这里应该是真实的数据加载器
    # for epoch in range(num_epochs):
    #     for batch in dataloader:
    #         loss = trainer.train_step(batch, loss_fn)
    #         
    #         if step % validation_interval == 0:
    #             # 验证等变性
    #             eq_results = trainer.validate_equivariance(batch)
    #             print(f"Equivariance check: {eq_results}")
    
    print("LoRA fine-tuning completed!")
    
    return model, trainer

if __name__ == "__main__":
    # 运行示例
    model = create_lora_model_example()
    ##引入新数据 按照 data.name 确定mof，进行其余数据的添加

    # 创建虚拟数据进行测试
    batch_size = 2
    num_atoms = 10
    
    class MockData:
        def __init__(self):
            self.natoms = [num_atoms] * batch_size
            self.pos = torch.randn(batch_size * num_atoms, 3)
            self.atomic_numbers = torch.randint(1, 19, (batch_size * num_atoms,))
            self.batch = torch.repeat_interleave(
                torch.arange(batch_size), torch.tensor(self.natoms)
            )
            self.energy = torch.randn(batch_size)
            self.forces = torch.randn(batch_size * num_atoms, 3)
    
    # 测试前向传播
    data = MockData()
    try:
        output = model(data)
        print("Forward pass successful!")
        print(f"Energy shape: {output['energy'].shape}")
        if 'forces' in output:
            print(f"Forces shape: {output['forces'].shape}")
        
        # 测试等变性
        validator = EquivarianceValidator()
        eq_results = validator.test_rotation_equivariance(model, data)
        print(f"Rotation equivariance test: {eq_results}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

