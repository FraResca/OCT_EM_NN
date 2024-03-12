from trainer import *

train_hybrid_multi_imbalance(15, True)
eval_hybrid_multi('hybrid_model_imbalance.h5')
visualize_attention('hybrid_model_imbalance.h5')