import torch
import os
import time
import logging


# paths
dataset_dir = '../dataset_v2'

# if not os.path.exists(dataset_dir):
#     raise Exception('Dataset directory not exist.')

train_code_path = os.path.join(dataset_dir, 'train.code')
train_sbt_path = os.path.join(dataset_dir, 'train.sbt')
train_nl_path = os.path.join(dataset_dir, 'train.comment')
train_ast_path = os.path.join(dataset_dir, 'train/train_ast.json')

valid_code_path = os.path.join(dataset_dir, 'valid.code')
valid_sbt_path = os.path.join(dataset_dir, 'valid.sbt')
valid_nl_path = os.path.join(dataset_dir, 'valid.comment')
valid_ast_path = os.path.join(dataset_dir, 'valid/valid_ast.json')

test_code_path = os.path.join(dataset_dir, 'test.code')
test_sbt_path = os.path.join(dataset_dir, 'test.sbt')
test_nl_path = os.path.join(dataset_dir, 'test.comment')
test_ast_path = os.path.join(dataset_dir, 'test/test_ast.json')

model_dir = 'model/'
best_model_path = 'best_model.pt'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

vocab_dir = 'vocab/'
code_vocab_path = 'code_vocab.pk'
ast_vocab_path = 'ast_vocab.pk'
nl_vocab_path = 'nl_vocab.pk'

code_vocab_txt_path = 'code_vocab.txt'
ast_vocab_txt_path = 'ast_vocab.txt'
nl_vocab_txt_path = 'nl_vocab.txt'

if not os.path.exists(vocab_dir):
    os.makedirs(vocab_dir)

out_dir = 'out/'    # other outputs dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# logger
log_dir = 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(os.path.join(log_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime())) + '.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# device
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# features
trim_vocab_min_count = False
trim_vocab_max_size = True

use_coverage = False
use_pointer_gen = False
use_teacher_forcing = True
use_check_point = False
use_lr_decay = True
use_early_stopping = True

validate_during_train = True
save_valid_model = True
save_best_model = True
save_test_details = True


# limitations
max_code_length = 313
max_nl_length = 30
min_nl_length = 4
max_decode_steps = 30
early_stopping_patience = 5
early_stopping_patience_meta = 5


# hyperparameters
vocab_min_count = 5
code_vocab_size = 50000  # 30000
nl_vocab_size = 30000    # 30000

learning_rate=0.001
embedding_dim = 256
hidden_size = 256
decoder_dropout_rate = 0.5
teacher_forcing_ratio = 0.5
batch_size = 32     # 128
code_encoder_lr = 0.001
ast_encoder_lr = 0.001
reduce_hidden_lr = 0.001
decoder_lr = 0.0001
lr_decay_every = 1
lr_decay_rate = 0.99
n_epochs = 50    # 50
support_batch_size=16
query_batch_size=16
beam_width = 5
beam_top_sentences = 1     # number of sentences beam decoder decode for one input
eval_batch_size = 32    # 16
test_batch_size = 16
init_uniform_mag = 0.02
init_normal_std = 1e-4


# visualization and resumes
print_every = 200  # 1000
plot_every = 10     # 100
save_model_every = 20   # 2000
save_check_point_every = 10   # 1000
validate_every = 5000     # 2000


# save config to log
save_config = True

config_be_saved = ['dataset_dir', 'use_cuda', 'device', 'use_coverage', 'use_pointer_gen', 'use_teacher_forcing',
                   'use_lr_decay', 'use_early_stopping', 'max_code_length', 'max_nl_length', 'min_nl_length',
                   'max_decode_steps', 'early_stopping_patience']

train_config_be_saved = ['embedding_dim', 'hidden_size', 'decoder_dropout_rate', 'teacher_forcing_ratio',
                         'batch_size', 'code_encoder_lr', 'ast_encoder_lr', 'reduce_hidden_lr',
                         'decoder_lr', 'lr_decay_every', 'lr_decay_rate', 'n_epochs']

eval_config_be_saved = ['beam_width', 'beam_top_sentences', 'eval_batch_size', 'test_batch_size']

if save_config:
    config_dict = locals()
    logger.info('Configurations this run are shown below.')
    logger.info('Notes: If only runs test, the model configurations shown above is not ' +
                'the configurations of the model test runs on.')
    logger.info('')
    logger.info('Features and limitations:')
    for config in config_be_saved:
        logger.info('{}: {}'.format(config, config_dict[config]))
    logger.info('')
    logger.info('Train configurations:')
    for config in train_config_be_saved:
        logger.info('{}: {}'.format(config, config_dict[config]))
    logger.info('')
    logger.info('Eval and test configurations:')
    for config in eval_config_be_saved:
        logger.info('{}: {}'.format(config, config_dict[config]))
    logger.info('')