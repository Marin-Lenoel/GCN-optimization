from shared.loaders import train_and_evaluate
from .MTGNN import MTGNN
from .GraphSAGE import GraphSAGE
from .RepeatLast import RepeatLast
from .GraphSAINT import GraphSAINT
from .ClusterGCN import ClusterGCN

MODELS = {
    "MTGNN": MTGNN,
    "GraphSAGE": GraphSAGE,
    "GraphSAGE-gru": GraphSAGE,
    "GraphSAGE-lstm": GraphSAGE,
    "GraphSAINT-gru": GraphSAINT,
    "GraphSAINT": GraphSAINT,
    "GraphSAINT-lstm": GraphSAINT,
    "RepeatLast": RepeatLast,
}

__all__ = [
    "MTGNN",
    "GraphSAGE",
    "RepeatLast",
    "GraphSAINT",
    "ClusterGCN",
    "MODELS",
]