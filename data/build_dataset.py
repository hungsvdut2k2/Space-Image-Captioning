from .data_normalize import text_normalize


class BuildDataset:
    def __init__(self, caption_file_path: str, max_sequence_length: int):
        self.caption_file_path = caption_file_path
        self.max_sequence_length = max_sequence_length
        self.dataset_dict = {}
        self.corpus = []

    def init_dataset(self):
        with open(self.caption_file_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n")
                img_path, caption = line.split("\t")

                tokens = caption.split(" ")

                if len(tokens) < 5 or len(tokens) > (self.max_sequence_length - 2):
                    continue
                else:
                    normalized_caption = text_normalize(caption)
                    self.corpus.append(normalized_caption)
                    self.dataset_dict[img_path] = normalized_caption
