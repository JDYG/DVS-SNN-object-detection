#--------------------------------------------------
#-----------------EMBEDDING------------------------
#--------------------------------------------------

from GEN1_od.models_spiking.embedding import EventEmbedding_NoPadding
# from GEN1_od.models_spiking.embedding_quan import EventEmbedding_NoPadding_Quan

EMBEDDINGs = {
    'EventEmbedding_NoPadding': EventEmbedding_NoPadding,
    # 'EventEmbedding_NoPadding_Quan': EventEmbedding_NoPadding_Quan
}

def build_embedding(param):
    param = param.copy()
    name = param.pop('type')
    cls = EMBEDDINGs[name]
    return cls(param)


#--------------------------------------------------
#-----------------ATTENTION------------------------
#--------------------------------------------------
from GEN1_od.models_spiking.attention import SparseOneWindowAttention_NoPadding
# from GEN1_od.models_spiking.attention_quan import SparseOneWindowAttention_NoPadding_Quan
from GEN1_od.models_spiking.attention_snn import SparseOneWindowAttention_NoPadding as SparseOneWindowAttention_NoPadding_SNN
ATTENTIONs = {
    'SparseOneWindowAttention_NoPadding': SparseOneWindowAttention_NoPadding,
    # 'SparseOneWindowAttention_NoPadding_Quan': SparseOneWindowAttention_NoPadding_Quan
    'SparseOneWindowAttention_NoPadding_SNN': SparseOneWindowAttention_NoPadding_SNN
}
def build_attention(param):
    param = param.copy()
    name = param.pop('type')
    cls = ATTENTIONs[name]
    return cls(param)



#--------------------------------------------------
#-----------------LATENT MEMORY------------------------
#--------------------------------------------------
from GEN1_od.models_spiking.latent_memory import Latent_Memory
# from GEN1_od.models_spiking.latent_memory_quan import Latent_Memory_Quan

LATENT_MEMORYs = {
    'Latent_Memory': Latent_Memory,
    # 'Latent_Memory_Quan': Latent_Memory_Quan

}
def build_latent_mem(param):
    param = param.copy()
    name = param.pop('type')
    cls = LATENT_MEMORYs[name]
    return cls(param)


#--------------------------------------------------
#-----------------DETECTION------------------------
#--------------------------------------------------
from GEN1_od.models_spiking.yolo import DetectionModel
DETECTIONs = {
    'Detection': DetectionModel
}
def build_detection(param):
    param = param.copy()
    name = param.pop('type')
    cls = DETECTIONs[name]
    return cls(param)


