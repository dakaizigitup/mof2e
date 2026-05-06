from fairchem.core.datasets import LmdbDataset
from fairchem.core.common.utils import setup_logging
from torch.utils.data import DataLoader
import torch

mof_name_list = [
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

# 设置日志
setup_logging()

def load_with_fairchem_dataset(mof_name_list):
    """使用 FairChem 的 LmdbDataset 加载数据"""
    
    # 配置参数
    config = {
        "src": "/data0/wfz/datasets/is2r/train/",  # LMDB 文件所在目录
        "normalize_labels": False,  # 是否标准化标签
        "target_mean": 0.0,
        "target_std": 1.0,
        "lin_ref": None,  # 线性参考能量
        "filter_mof_names": mof_name_list  # 添加过滤 MOF 名称的参数
    }
    
    try:
        # 创建数据集
        dataset = LmdbDataset(config)
        print(f"✅ 数据集加载成功")
        print(f"数据集大小: {len(dataset)}")
        defect_list = []
        for i in range(len(dataset)):
            # 获取第一个样本
            sample = dataset[i]
            mof_name = sample['name'].split('_')[0]
            
            # print(mof_name)
            if mof_name in mof_name_list:
                defect_name = sample['name']
                defect_list.append(defect_name)
                
        print(defect_list)
        # 保存 defect_list 为 txt 文件
        with open('defect_list.txt', 'w', encoding='utf-8') as f:
            for defect_name in defect_list:
                f.write(defect_name + '\n')
        
        print(f"✅ defect_list 已保存到 defect_list.txt 文件，共 {len(defect_list)} 条记录")
                
            

        
        
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

# 使用示例

dataset = load_with_fairchem_dataset(mof_name_list)
