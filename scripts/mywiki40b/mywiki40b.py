#!/usr/bin/env python3
import gzip
from pathlib import Path
from typing import Optional

import tensorflow_datasets as tfds

_DESCRIPTION = """
Mywiki40b corpus
"""

_CITATION = """
"""


DATA_SOURCE: Optional[str] = None


def generate_examples_from_file(f: Path):
    with gzip.open(f, mode='rt') as txtf:
        for lid, line in enumerate(txtf):
            yield f'mywiki40b-{lid:10}', {'text': line.strip()}


class Mywiki40b(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mywiki40b dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
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
