from trainer import *

train_hybrid_multi_imbalance_noruler(15, True)
eval_hybrid_multi('hybrid_model_imbalance_noruler.h5')
visualize_attention_noruler('hybrid_model_imbalance_noruler.h5')