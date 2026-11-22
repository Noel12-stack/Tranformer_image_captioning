"""Unit tests for dataset loading, tokenization, and vocabulary."""

import unittest
import torch
from dataset.vocabulary import Vocabulary
from dataset.preprocessing import clean_caption, tokenize_caption
from dataset.dataset import create_synthetic_dataset, Flickr8kDataset, CaptionCollate
from dataset.split_dataset import parse_captions_file


class TestDatasetAndVocabulary(unittest.TestCase):

    def test_caption_cleaning_and_tokenization(self):
        raw = "A Dog, running in the green-field... #1!"
        cleaned = clean_caption(raw)
        self.assertEqual(cleaned, "a dog running in the green field 1")
        tokens = tokenize_caption(raw)
        self.assertEqual(tokens, ["a", "dog", "running", "in", "the", "green", "field", "1"])

    def test_vocabulary_special_tokens(self):
        vocab = Vocabulary(min_freq=1)
        self.assertEqual(vocab.w2i["<PAD>"], 0)
        self.assertEqual(vocab.w2i["<SOS>"], 1)
        self.assertEqual(vocab.w2i["<EOS>"], 2)
        self.assertEqual(vocab.w2i["<UNK>"], 3)
        self.assertEqual(vocab.PAD_IDX, 0)
        self.assertEqual(vocab.SOS_IDX, 1)
        self.assertEqual(vocab.EOS_IDX, 2)
        self.assertEqual(vocab.UNK_IDX, 3)

    def test_vocabulary_build_and_numericalize(self):
        captions = ["a cute brown dog", "a small black cat", "a brown dog running"]
        vocab = Vocabulary(min_freq=2)
        vocab.build_vocabulary(captions)

        # Words with freq >= 2: 'a' (3), 'brown' (2), 'dog' (2)
        self.assertIn("a", vocab.w2i)
        self.assertIn("brown", vocab.w2i)
        self.assertIn("dog", vocab.w2i)
        # 'cat' has freq 1 -> should map to UNK
        self.assertNotIn("cat", vocab.w2i)

        encoded = vocab.numericalize("a cat dog")
        self.assertEqual(encoded[0], vocab.SOS_IDX)
        self.assertEqual(encoded[-1], vocab.EOS_IDX)
        self.assertEqual(encoded[1], vocab.w2i["a"])
        self.assertEqual(encoded[2], vocab.UNK_IDX)  # cat -> UNK
        self.assertEqual(encoded[3], vocab.w2i["dog"])

        decoded = vocab.decode(encoded)
        self.assertEqual(decoded, "a dog")  # UNK omitted by default

    def test_synthetic_dataset_and_collate(self):
        images_dir, captions_file = create_synthetic_dataset("data/test_tmp", num_images=4)
        mapping = parse_captions_file(captions_file)
        self.assertEqual(len(mapping), 4)

        vocab = Vocabulary(min_freq=1)
        all_caps = [c for caps in mapping.values() for c in caps]
        vocab.build_vocabulary(all_caps)

        ds = Flickr8kDataset(images_dir, list(mapping.keys()), mapping, vocab)
        self.assertGreater(len(ds), 0)

        img, tokens, name = ds[0]
        self.assertEqual(img.shape, (3, 224, 224))
        self.assertIsInstance(tokens, torch.Tensor)

        collate = CaptionCollate(pad_idx=vocab.PAD_IDX)
        batch = [ds[0], ds[1]]
        imgs, padded_caps, names = collate(batch)
        self.assertEqual(imgs.shape, (2, 3, 224, 224))
        self.assertEqual(padded_caps.dim(), 2)
        self.assertEqual(padded_caps.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
