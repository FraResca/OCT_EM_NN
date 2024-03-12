from trainer import *

train_hybrid_multi(15, True)
eval_hybrid_multi('hybrid_model.h5')
visualize_attention('hybrid_model.h5')