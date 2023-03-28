import torch


class TextDataset(torch.utils.data.Dataset):
    """Stores lines from a text file."""
    def __init__(self, file_path, transform, portion=1):
        """
        Reads lines from `file_path`, applies `transform` and saves both the raw and transformed lines to memory.
        If a `portion` between 0 and 1 is specified, only stores that portion of the lines in the file
        (from the beginning of the file).
        """
        super().__init__()
        assert 0 <= portion <= 1
        self.raw_lines = file_path.read_text().split('\n')
        self.raw_lines = self.raw_lines[:int(len(self.raw_lines) * portion)]
        self.data = [transform(line) for line in self.raw_lines]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class ZipDataset(torch.utils.data.Dataset):
    """Zips multiple datasets of the same length analogously to the built-in zip() function."""
    def __init__(self, *datasets):
        super().__init__()
        assert datasets and all(len(ds) == len(datasets[0]) for ds in datasets[1:])
        self.datasets = datasets
    
    def __getitem__(self, item):
        return tuple(ds[item] for ds in self.datasets)
    
    def __len__(self):
        return len(self.datasets[0])
