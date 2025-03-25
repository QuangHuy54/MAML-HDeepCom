from torch.utils.data import Dataset

import utils


class CodePtrDataset(Dataset):

    def __init__(self, code_path, ast_path, nl_path,num_of_data=-1,seed=1):
        # get lines
        codes = utils.load_dataset(code_path,num_of_data,seed)
        asts = utils.load_dataset(ast_path,num_of_data,seed)
        nls = utils.load_dataset(nl_path,num_of_data,seed)

        if len(codes) != len(asts) or len(codes) != len(nls) or len(asts) != len(nls):
            raise Exception('The lengths of three dataset do not match.')

        self.codes, self.asts, self.nls = utils.filter_data(codes, asts, nls)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.codes[index], self.asts[index], self.nls[index]

    def get_dataset(self):
        return self.codes, self.asts, self.nls
