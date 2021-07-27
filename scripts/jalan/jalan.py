#!/usr/bin/env python3
import gzip
import unicodedata
from pathlib import Path
from typing import List, Optional

import tensorflow_datasets as tfds

_DESCRIPTION = """
Jalan corpus
"""

_CITATION = """
"""


DATA_SOURCE: Optional[str] = None


def generate_examples_from_file(f: Path):
    docid: str = ''
    prevdocid = None
    sents: List[str] = []
    with gzip.open(f, mode='rt') as txtf:
        for line in txtf:
            if line.startswith('# S-ID'):
                sid = line[7:]
                docid = sid.split('-')[0]
                if docid != prevdocid and len(sents) > 0:
                    yield docid, {'text': ''.join(sents)}
                    sents = []
                prevdocid = docid
            else:
                sents.append(
                    unicodedata.normalize('NFKC', line.strip()))

    if docid != prevdocid and len(sents) > 0:
        yield docid, {'text': ''.join(sents)}


class Jalan(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for jalan dataset."""

    VERSION = tfds.core.Version('1.0.1')
    RELEASE_NOTES = {
        '1.0.1': 'Initial release',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "text": tfds.features.Text(), }),
            supervised_keys=None,
            homepage="https://dummy.invalid",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        assert DATA_SOURCE is not None
        return {
            'train': self._generate_examples(Path(DATA_SOURCE)),
        }

    def _generate_examples(self, path: Path):
        for f in sorted(path.iterdir()):
            yield from generate_examples_from_file(f)
