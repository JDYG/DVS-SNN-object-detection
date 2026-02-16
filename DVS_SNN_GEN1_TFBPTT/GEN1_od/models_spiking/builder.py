#--------------------------------------------------
#-----------------EMBEDDING------------------------
#--------------------------------------------------

from GEN1_od.models_spiking.embedding import EventEmbedding_NoPadding
from GEN1_od.models_spiking.embedding_quan import EventEmbedding_NoPadding_Quan
EMBEDDINGs = {
    'EventEmbedding_NoPadding': EventEmbedding_NoPadding,
    'EventEmbedding_NoPadding_Quan': EventEmbedding_NoPadding_Quan
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

ATTENTIONs = {
    'SparseOneWindowAttention_NoPadding': SparseOneWindowAttention_NoPadding,

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

LATENT_MEMORYs = {
    'Latent_Memory': Latent_Memory,

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
