class RecursiveSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.symbols = ["\n\n", "\n", " ", ""]

    def symbol_splits(
        self, text: str, split_level: int
    ) -> tuple[list[str], int]:
        cur_sep = self.symbols[split_level]
        if text.find(cur_sep) == -1:
            return self.symbol_splits(text, split_level + 1)
        if split_level == len(self.symbols) - 1:
            return ([s for s in text], split_level)
        parts = text.split(cur_sep)
        splits = []
        for part in parts:
            if len(part) > 0:
                splits.append(part)
        return (splits, split_level)

    def merge_good_splits(
        self, good_splits: list[str], split_level: int
    ) -> list[str]:
        result_splits = []
        current_chunk = []
        current_len = 0
        sep = self.symbols[split_level]
        sep_len = len(sep)

        for split in good_splits:
            add_len = len(split) + (sep_len if current_chunk else 0)

            if current_len + add_len > self.chunk_size:
                if current_chunk:
                    chunk_str = sep.join(current_chunk)
                    result_splits.append(chunk_str)
                    overlap_chunk = []
                    overlap_len = 0
                    for i in range(len(current_chunk)-1, -1, -1):
                        frag = current_chunk[i]
                        frag_add_len = len(frag) + \
                            (sep_len if overlap_chunk else 0)
                        if overlap_len + frag_add_len <= self.chunk_overlap:
                            overlap_chunk.insert(0, frag)
                            overlap_len += frag_add_len
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_len = overlap_len
                else:
                    current_chunk = [split]
                    current_len = len(split)
                    continue

            current_chunk.append(split)
            current_len += add_len
        if current_chunk:
            result_splits.append(sep.join(current_chunk))
        return result_splits

    def merge_splits(self, splits: list[str], split_level: int) -> list[str]:
        good_splits = []
        final_splits = []
        for split in splits:
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                if len(good_splits) > 0:
                    final_splits.extend(
                        self.merge_good_splits(good_splits, split_level))
                    good_splits = []
                final_splits.extend(
                    self.merge_splits(
                        *self.symbol_splits(split, split_level + 1))
                )
        if good_splits:
            final_splits.extend(
                self.merge_good_splits(good_splits, split_level))
        return final_splits

    def split_text(self, text: str) -> list[str]:
        chunks = []
        init_splits, init_split_level = self.symbol_splits(text, 0)
        chunks = self.merge_splits(
            init_splits, init_split_level
        )
        return [chunk for chunk in chunks if len(chunk) > 0]
