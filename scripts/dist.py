#!/usr/bin/env python3

import argparse
from pathlib import Path

import tensorflow.io.gfile
from transformers import T5Tokenizer


def save_tokenizer(path_in: Path, path_out: Path) -> None:
    tokenizer = T5Tokenizer(str(path_in))
    tokenizer.save_pretrained(str(path_out))


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", required=True)
    oparser.add_argument("--tokenizer", "-t", required=True)
    oparser.add_argument("--output", "-o", type=Path, required=True)
    oparser.add_argument("--config", "-c", required=True)
    oparser.add_argument("--nocopy", action='store_true')
    return oparser.parse_args()


def get_model_name(path_out_tf_misc) -> str:
    with tensorflow.io.gfile.GFile(path_out_tf_misc.joinpath('checkpoint')) as cf:
        for line in cf:
            if line.startswith('model_checkpoint_path: '):
                return line[len('model_checkpoint_path: "'):-2]
    raise KeyError


def copy(path_in: str, path_out_tf: Path, path_spm: Path, path_tokenizer: Path):
    path_in = path_in.rstrip('/')
    path_out_tf_misc: Path = path_out_tf.joinpath('misc')
    path_out_tf_misc.mkdir(parents=True, exist_ok=True)

    print('Copying...')
    tensorflow.io.gfile.copy(path_tokenizer, path_spm, overwrite=True)

    for target in ['operative_config.gin', 'graph.pbtxt', 'checkpoint']:
        tensorflow.io.gfile.copy(f'{path_in}/{target}', path_out_tf_misc.joinpath(target), overwrite=True)
    for f in tensorflow.io.gfile.glob(f'{path_in}/events.out.**'):
        tensorflow.io.gfile.copy(f, path_out_tf_misc.joinpath(Path(f).name), overwrite=True)

    model_name: str = get_model_name(path_out_tf_misc)
    print(f'Model name: <{model_name}>')
    for f in tensorflow.io.gfile.glob(f'{path_in}/{model_name}*'):
        print(f'Copying {f}...')
        tensorflow.io.gfile.copy(f, path_out_tf.joinpath(Path(f).name), overwrite=True)
    return path_out_tf.joinpath(model_name)


def main() -> None:
    opts = get_opts()

    path_out_tf: Path = opts.output.joinpath('tf')
    path_out_tf.mkdir(parents=True, exist_ok=True)
    path_spm = path_out_tf.joinpath(Path(opts.tokenizer).name)

    path_out_pt: Path = opts.output.joinpath('pt')
    path_out_pt.mkdir(parents=True, exist_ok=True)

    if opts.nocopy:
        path_out_tf = Path(opts.input)
        path_out_tf_misc: Path = path_out_tf.joinpath('misc')
        model_name: str = get_model_name(path_out_tf_misc)
        path_tf_model = path_out_tf.joinpath(model_name)
    else:
        path_tf_model = copy(opts.input, path_out_tf, path_spm, opts.tokenizer)

    print('Converting spm...')
    save_tokenizer(path_spm, path_out_pt)

    print('Converting model...')
    from transformers.models.t5.convert_t5_original_tf_checkpoint_to_pytorch import \
        convert_tf_checkpoint_to_pytorch
    convert_tf_checkpoint_to_pytorch(
        tf_checkpoint_path=path_tf_model,
        config_file=opts.config,
        pytorch_dump_path=path_out_pt,
    )


if __name__ == '__main__':
    main()
