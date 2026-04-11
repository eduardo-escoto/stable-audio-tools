import json
import os
import random

LOOKUP_PATH = os.environ.get(
    "MIN_NOISE_LEVEL_LOOKUP",
    "/deepfreeze/pnlong/ambient/spdmx_min_noise_levels.json",
)

_lookup: dict = None


def _load_lookup():
    global _lookup
    if _lookup is None:
        if os.path.isfile(LOOKUP_PATH):
            with open(LOOKUP_PATH) as f:
                _lookup = json.load(f)
        else:
            _lookup = {}


def get_custom_metadata(info, audio):
    _load_lookup()

    prompt = info['prompt']
    # randomly shuffle the sentences in the prompt, and drop each with 0.1 probability
    sentences = prompt.split('. ')
    random.shuffle(sentences)
    sentences = [s for s in sentences if random.random() > 0.1]
    caption = '. '.join(sentences)

    result = {"prompt": caption}

    mnl = _lookup.get(info.get("latent_filename", ""), None)
    if mnl is not None:
        result["min_noise_level"] = mnl

    return result
