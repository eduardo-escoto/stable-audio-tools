import random
import sys
from pathlib import Path

_cm_dir = str(Path(__file__).resolve().parent)
if _cm_dir not in sys.path:
    sys.path.insert(0, _cm_dir)


def get_custom_metadata(info, audio):
    
    prompt = info['prompt']
    # randomly shuffle the sentences in the prompt, and drop each with 0.1 probability
    sentences = prompt.split('. ')
    random.shuffle(sentences)
    sentences = [s for s in sentences if random.random() > 0.1]
    caption = '. '.join(sentences)

    return {
        "prompt": caption,
    }