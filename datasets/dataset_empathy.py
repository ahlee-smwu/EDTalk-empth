"""Dataset for empathetic talking-head training.

Each sample returns a **speaker** frame (and optionally mel-spectrogram)
together with a **listener** identity frame and listener target frame.
The listener target embodies the ground-truth empathetic reaction to the
speaker.

The dataset is backed by an LMDB store and a JSON list file whose
entries follow the convention:

    ``<speaker_id>#<listener_id>#<emotion>#<clip_id>``

When such four-part keys are not available the dataset falls back to
treating every video as a self-reenactment pair (speaker == listener)
so that the code can also run on MEAD/HDTF data without a dedicated
empathy annotation.
"""

import os
import json
import lmdb
import random
import collections
import numpy as np
from io import BytesIO
from PIL import Image

import torch
from torch.utils.data import Dataset


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


class EmpathyDataset(Dataset):
    """Speaker–listener paired dataset for empathy training.

    Args:
        opt: Namespace with at least ``path`` (LMDB root), ``resolution``.
        is_inference: if True load the test split.
        transform: torchvision transform applied to every image.
        list_train: path to training JSON list.
        list_test:  path to test JSON list.
    """

    def __init__(
        self,
        opt,
        is_inference=False,
        transform=None,
        list_train='lists/empathy_train.json',
        list_test='lists/empathy_test.json',
    ):
        self.env = lmdb.open(
            opt.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', opt.path)

        list_file = list_test if is_inference else list_train
        with open(list_file, 'r') as f:
            videos = json.load(f)

        self.resolution = opt.resolution
        self.transform = transform

        # Build index structures -----------------------------------------------
        self.pairs = []          # list of (speaker_video_item, listener_video_item)
        self.pair_keys = []      # string key for grouping
        self._build_index(videos)

    # ------------------------------------------------------------------
    def _build_index(self, videos):
        """Parse JSON entries into speaker/listener video pairs."""
        speaker_items = collections.defaultdict(list)
        listener_items = collections.defaultdict(list)

        for entry in videos:
            parts = entry.split('#')
            item = self._make_video_item(entry)
            if item is None:
                continue
            if len(parts) >= 4:
                # Convention: speaker_id#listener_id#emotion#clip_id
                spk_key = parts[0]
                lis_key = parts[1]
                speaker_items[spk_key].append(item)
                listener_items[lis_key].append(item)
            else:
                # Fallback: self-reenactment (same person as both roles)
                person_key = parts[0]
                speaker_items[person_key].append(item)
                listener_items[person_key].append(item)

        # Pair up every shared key
        shared_keys = set(speaker_items.keys()) & set(listener_items.keys())
        for key in shared_keys:
            for s_item in speaker_items[key]:
                for l_item in listener_items[key]:
                    self.pairs.append((s_item, l_item))
                    self.pair_keys.append(key)

        # If no pairs were formed fall back to random cross-pairing
        if len(self.pairs) == 0:
            all_items = []
            for entry in videos:
                item = self._make_video_item(entry)
                if item is not None:
                    all_items.append(item)
            for item in all_items:
                self.pairs.append((item, item))
                self.pair_keys.append(item['video_name'])

    def _make_video_item(self, video_name):
        item = {'video_name': video_name}
        try:
            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(video_name, 'length')
                length = int(txn.get(key).decode('utf-8'))
            item['num_frame'] = length
            return item
        except Exception:
            return None

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        speaker_item, listener_item = self.pairs[index]

        # Random frames
        spk_frame_idx = random.randint(0, speaker_item['num_frame'] - 1)
        lis_src_idx, lis_tgt_idx = sorted(
            random.sample(range(listener_item['num_frame']),
                          k=min(2, listener_item['num_frame']))
        )
        if listener_item['num_frame'] < 2:
            lis_tgt_idx = lis_src_idx

        with self.env.begin(write=False) as txn:
            spk_bytes = txn.get(
                format_for_lmdb(speaker_item['video_name'], spk_frame_idx)
            )
            lis_src_bytes = txn.get(
                format_for_lmdb(listener_item['video_name'], lis_src_idx)
            )
            lis_tgt_bytes = txn.get(
                format_for_lmdb(listener_item['video_name'], lis_tgt_idx)
            )

        speaker_img = Image.open(BytesIO(spk_bytes))
        listener_src_img = Image.open(BytesIO(lis_src_bytes))
        listener_tgt_img = Image.open(BytesIO(lis_tgt_bytes))

        if self.transform is not None:
            speaker_img = self.transform(speaker_img)
            listener_src_img = self.transform(listener_src_img)
            listener_tgt_img = self.transform(listener_tgt_img)

        data = {
            'speaker_frame': speaker_img,
            'listener_source': listener_src_img,
            'listener_target': listener_tgt_img,
        }
        return data
