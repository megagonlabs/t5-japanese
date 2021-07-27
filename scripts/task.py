#!/usr/bin/env python3


import functools
import os

import seqio
import t5.data
from tensorflow.python.tpu import tpu_system_metadata

tpu_system_metadata._RETRY_TIMES = 2

DEFAULT_SPM_PATH = os.environ['SPM']
DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(DEFAULT_SPM_PATH)
DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True)
}


task_names = ["jalan", "mywiki40b"]
data_names = ["jalan", "mywiki40b"]
versions = ["1.0.1", "1.0.0"]


for task_name, data_name, version in zip(task_names, data_names, versions):
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TfdsDataSource(
            tfds_name=f"{data_name}:{version}",
            splits={"train": "train"},
        ),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.rekey,
                key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            t5.data.preprocessors.span_corruption,
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[])


seqio.MixtureRegistry.add(task_name, [(task_name, 1.0)])


lang: str = "ja"
seqio.TaskRegistry.add(f"mc4.{lang}",
                       source=seqio.TfdsDataSource(
                           tfds_name="c4/multilingual:3.0.1",
                           splits={
                               "train": lang,
                               "validation": f"{lang}-validation"
                           }),
                       preprocessors=[
                           functools.partial(
                               t5.data.preprocessors.rekey,
                               key_map={
                                   "inputs": None,
                                   "targets": "text"
                               }),
                           seqio.preprocessors.tokenize,
                           seqio.CacheDatasetPlaceholder(),
                           t5.data.preprocessors.span_corruption,
                           seqio.preprocessors.append_eos_after_trim,
                       ],
                       output_features=DEFAULT_OUTPUT_FEATURES,
                       metric_fns=[])
seqio.MixtureRegistry.add("mc4", [(f"mc4.{lang}", 1.0)])


# Mixture of mC4 and WIKI

DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)
seqio.MixtureRegistry.add(
    "mc4_mywiki40b",
    [f"mc4.{lang}", "mywiki40b"],
    default_rate=DEFAULT_MIX_RATE,
)
seqio.MixtureRegistry.add(
    "jalan_mywiki40b",
    ["jalan", "mywiki40b"],
    default_rate=DEFAULT_MIX_RATE,
)
