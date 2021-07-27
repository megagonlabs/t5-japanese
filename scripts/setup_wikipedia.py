#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Iterator

import tensorflow_datasets as tfds
import tqdm


def extract_wikipedia(tf_data) -> Iterator[str]:
    in_paragraph: bool = False

    for wiki in tqdm.tqdm(tf_data.as_numpy_iterator()):
        for text in wiki['text'].decode().split('\n'):
            if in_paragraph is True:
                yield text.replace('_NEWLINE_', '')
                in_paragraph = False
            if text == '_START_PARAGRAPH_':
                in_paragraph = True


def operation(path_in: Path, path_out: Path, lang: str) -> None:
    with path_out.open('w') as outf:
        for split in ['validation', 'test', 'train']:
            ds = tfds.load(f'wiki40b/{lang}', split=split)
            for line in extract_wikipedia(ds):
                outf.write(line)
                outf.write('\n')


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path,
                         default='/dev/stdin', required=False)
    oparser.add_argument("--output", "-o", type=Path,
                         default="/dev/stdout", required=False)
    oparser.add_argument("--lang", default="ja")
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    operation(opts.input, opts.output, opts.lang)


if __name__ == '__main__':
    main()
