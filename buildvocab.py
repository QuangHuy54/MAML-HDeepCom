import utils
import config
import data

code_path='../dataset_v2/original/all.code'
ast_path='../dataset_v2/original/all.sbt'
nl_path='../dataset_v2/original/all.comment'
dataset = data.CodePtrDataset(code_path,ast_path,nl_path)
code_vocab: utils.Vocab
ast_vocab: utils.Vocab
nl_vocab: utils.Vocab
code_vocab = utils.Vocab('code_vocab')
ast_vocab = utils.Vocab('ast_vocab')
nl_vocab = utils.Vocab('nl_vocab')
codes, asts, nls = dataset.get_dataset()
for code, ast, nl in zip(codes, asts, nls):
    code_vocab.add_sentence(code)
    ast_vocab.add_sentence(ast)
    nl_vocab.add_sentence(nl)

origin_code_vocab_size = len(code_vocab)
origin_nl_vocab_size = len(nl_vocab)

# trim vocabulary
code_vocab.trim(config.code_vocab_size)
nl_vocab.trim(config.nl_vocab_size)
# save vocabulary
code_vocab.save(config.code_vocab_path)
ast_vocab.save(config.ast_vocab_path)
nl_vocab.save(config.nl_vocab_path)
code_vocab.save_txt(config.code_vocab_txt_path)
ast_vocab.save_txt(config.ast_vocab_txt_path)
nl_vocab.save_txt(config.nl_vocab_txt_path)

code_vocab_size = len(code_vocab)
ast_vocab_size = len(ast_vocab)
nl_vocab_size = len(nl_vocab)
print("Vocab code: ",code_vocab_size)
print("Vocab ast: ",ast_vocab_size)
print("Vocab nl: ",nl_vocab_size)
