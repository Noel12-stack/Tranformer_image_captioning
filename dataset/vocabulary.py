"""
Vocabulary Management Module for Image Captioning.

Handles:
- Special token indices: <PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3
- Frequency-based vocabulary filtering from training split only (prevents data leakage)
- Mapping words to integers (numericalization)
- Mapping integer predictions back to words (decoding)
- Serialization to and deserialization from JSON
"""

import json
from collections import Counter
from typing import List, Dict, Union
from .preprocessing import tokenize_caption


class Vocabulary:
    """
    Vocabulary class that manages word-to-index and index-to-word mappings.
    """

    PAD_TOKEN = "<PAD>"  # Padding token
    SOS_TOKEN = "<SOS>"  # Start of sequence
    EOS_TOKEN = "<EOS>"  # End of sequence
    UNK_TOKEN = "<UNK>"  # Unknown out-of-vocabulary word

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Initialize fixed special tokens."""
        self.w2i = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.i2w = {idx: token for token, idx in self.w2i.items()}

    def __len__(self) -> int:
        return len(self.w2i)

    def build_vocabulary(self, captions: List[str]) -> None:
        """
        Build vocabulary exclusively from training split captions.

        Args:
            captions: List of human-written caption strings.
        """
        self._init_special_tokens()
        counter = Counter()

        for cap in captions:
            tokens = tokenize_caption(cap)
            counter.update(tokens)

        # Filter words by minimum frequency threshold
        current_idx = len(self.w2i)
        for word, count in counter.most_common():
            if count >= self.min_freq and word not in self.w2i:
                self.w2i[word] = current_idx
                self.i2w[current_idx] = word
                current_idx += 1

    def numericalize(self, caption: Union[str, List[str]], max_len: int = None) -> List[int]:
        """
        Convert a caption string or token list into numerical IDs:
        [<SOS>, token_1, token_2, ..., <EOS>]

        Args:
            caption: Caption string or pre-split token list.
            max_len: Optional max length (including SOS and EOS).

        Returns:
            List of integer token IDs.
        """
        if isinstance(caption, str):
            tokens = tokenize_caption(caption)
        else:
            tokens = caption

        numerical = [self.SOS_IDX]
        for token in tokens:
            numerical.append(self.w2i.get(token, self.UNK_IDX))
        numerical.append(self.EOS_IDX)

        if max_len is not None and len(numerical) > max_len:
            # Truncate and ensure final token is EOS
            numerical = numerical[: max_len - 1] + [self.EOS_IDX]

        return numerical

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert a list of numerical token IDs back into a human-readable string.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: If True, omit <PAD>, <SOS>, <EOS>, <UNK>.

        Returns:
            Decoded caption string.
        """
        words = []
        special_indices = {self.PAD_IDX, self.SOS_IDX, self.EOS_IDX, self.UNK_IDX}

        for idx in token_ids:
            if idx == self.EOS_IDX and skip_special_tokens:
                break
            if skip_special_tokens and idx in special_indices:
                continue
            words.append(self.i2w.get(idx, self.UNK_TOKEN))

        return " ".join(words)

    def save(self, filepath: str) -> None:
        """Save vocabulary mappings to a JSON file."""
        data = {
            "min_freq": self.min_freq,
            "w2i": self.w2i,
            "i2w": {str(k): v for k, v in self.i2w.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Load vocabulary mappings from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(min_freq=data.get("min_freq", 2))
        vocab.w2i = data["w2i"]
        vocab.i2w = {int(k): v for k, v in data["i2w"].items()}
        return vocab
