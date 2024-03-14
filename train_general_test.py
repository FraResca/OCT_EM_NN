from trainer import *

train_general(20, True, True, True, 'hybrid_test')
eval_general('hybrid_test', True, True)
visualize_attention_general('hybrid_test', True, True)