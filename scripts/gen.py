#!/usr/bin/env python3

import argparse
from pathlib import Path

import tensorflow_datasets as tfds


def convert_jalan(path_in: Path, path_out: Path, version: str = '1.0.1'):
    import jalan  # noqa
    jalan.jalan.DATA_SOURCE = str(path_in)
    assert 'jalan' in tfds.list_builders()

    return tfds.load(name=f'jalan:{version}',
                     data_dir=path_out,
                     download=True,
                     #                             shuffle_files=True,
                     try_gcs=False,
                     with_info=True,
                     )


def convert_mywiki40b(path_in: Path, path_out: Path, version: str = '1.0.0'):
    import mywiki40b  # noqa
    mywiki40b.mywiki40b.DATA_SOURCE = str(path_in)
    assert 'mywiki40b' in tfds.list_builders()

    return tfds.load(name=f'mywiki40b:{version}',
                     data_dir=path_out,
                     download=True,
                     try_gcs=False,
                     with_info=True,
                     )


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, required=True)
    oparser.add_argument("--output", "-o", type=Path, required=True)
    oparser.add_argument("--source", "-s", required=True, choices=['jalan', 'mywiki40b'])
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    if opts.source == 'jalan':
        ds, ds_info = convert_jalan(opts.input, opts.output)
    elif opts.source == 'mywiki40b':
        ds, ds_info = convert_mywiki40b(opts.input, opts.output)
    else:
        raise NotImplementedError

    print(ds_info)

    train_ds = ds["train"].batch(10)
    batches = train_ds.take(1)
    for batch in tfds.as_numpy(batches):
        raw_texts = batch["text"]
        for i in range(batch["text"].shape[0]):
            print("%3d, %s" % (i, raw_texts[i].decode("utf-8")))


if __name__ == '__main__':
    main()
