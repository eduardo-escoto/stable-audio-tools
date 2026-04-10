"""Shared constants for custom metadata modules under this directory.

These files are often loaded via importlib (see stable_audio_tools.data.dataset);
import them using the path bootstrap pattern in each consumer module.
"""

MIN_NOISE_LEVEL = 0.7

# Wall-clock seconds for the `seconds_total` number conditioner (see PreEncodedDataset
# and model configs). Not latent frames: NumberConditioner normalizes this value using
# min_val/max_val from the conditioning config (e.g. 0..256 seconds).
CLIP_SECONDS_TOTAL = (256 * 2048) / 44100
