import sys
from pathlib import Path
from random import shuffle

from pandas import read_csv, isna

_cm_dir = str(Path(__file__).resolve().parent)
if _cm_dir not in sys.path:
    sys.path.insert(0, _cm_dir)

sPDMX_md = None

md_columns = [
    "genres",
    "groups",
    "tags",
    "song_name",
    "title",
    "subtitle",
    "artist_name",
    "composer_name",
    "publisher",
    "complexity",
    "license",
]


def get_custom_metadata(info, audio):
    global sPDMX_md
    spdmx_base_path = Path(info["path"]).parents[3]
    
    if sPDMX_md is None:
        csv_path = spdmx_base_path / "data.csv"
        sPDMX_md = read_csv(csv_path)

    # track_rel_path = "./" + str(Path(info["path"]).relative_to(spdmx_base_path))
    track_rel_path = "/deepfreeze/user_shares/pnlong/sPDMX_stems/" + str(Path(info["path"]).relative_to(spdmx_base_path))
    # print(track_rel_path)
    track_md = sPDMX_md[(sPDMX_md["path"] == track_rel_path)][md_columns]

    track_md_vals = [
        f"{k}: {i}"
        for k, i in dict(zip(track_md.columns, track_md.values[0])).items()
        if not isna(i)
    ]

    shuffle(track_md_vals)

    track_caption = ", ".join(track_md_vals)
    # print(track_caption)
    return {
        "prompt": track_caption,
    }
