import os
import json
from json import dump as json_dump

from einops import rearrange

import torch
import pytorch_lightning as pl
from safetensors.torch import save_file as sf_save_file
from stable_audio_tools import get_pretrained_model
from prefigure.prefigure import get_all_args
from stable_audio_tools.models import create_model_from_config
from pytorch_lightning.callbacks import Callback, BasePredictionWriter
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models.utils import (
    load_ckpt_state_dict,
)
from stable_audio_tools.training.utils import copy_state_dict

import torchaudio


def trim_to_shortest(a, b):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:, :, : b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:, :, : a.shape[-1]]
    return a, b


class PreEncodePipeline(pl.LightningModule):
    def __init__(
        self,
        pretransform,  # dataset,
        model_config,
        dataset_config,
    ):
        super().__init__()
        self.pretransform = pretransform
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.prediction_step_outputs = []

    def forward(self, x):
        raise NotImplementedError(
            "Pre-encoding is not trainable so forward is not needed"
        )

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Pre-encoding is not a training task")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Pre-encoding is not a validation task")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # self.print(f"Predicting batch {batch_idx} from dataloader {dataloader_idx}")
        # batches should have audio and info, where info has a bunch of metadata
        reals, infos = batch

        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if len(reals.shape) == 2:
            reals = reals.unsqueeze(1)

        pre_enc_info = {
            "infos": infos,
            "input_reals": reals,
        }

        encoder_input = self.pretransform.model.preprocess_audio_list_for_encoder(
            reals, self.model_config["sample_rate"]
        )
        pre_enc_info["processed_input_reals"] = encoder_input

        # if self.force_input_mono and encoder_input.shape[1] > 1:
        #     encoder_input = encoder_input.mean(dim=1, keepdim=True)

        with torch.no_grad():
            latents, encoder_info = self.pretransform.encode(
                encoder_input, return_info=True
            )
            pre_enc_info["latents"] = latents
            pre_enc_info["encoder_infos"] = encoder_info

            decoded = self.pretransform.decode(latents)
            # Trim output to remove post-padding.
            decoded, trim_reals = trim_to_shortest(decoded.clone(), reals.clone())

        pre_enc_info["decoded_reals"] = decoded
        pre_enc_info["trimmed_input_reals"] = trim_reals

        # self.prediction_step_outputs.append(pre_enc_info)

        return pre_enc_info


class PreEncodePredictionWriter(BasePredictionWriter):
    def __init__(self, output_path, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_path = output_path

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        infos = prediction["infos"]
        input_reals = prediction["input_reals"]
        processed_input_reals = prediction["processed_input_reals"]
        latents = prediction["latents"]
        decoded_reals = prediction["decoded_reals"]
        trimmed_input_reals = prediction["trimmed_input_reals"]
        encoder_infos = prediction["encoder_infos"]

        for idx, info in enumerate(infos):
            out_metadata_dict = {k: v for k, v in info.items() if k != "padding_mask"}
            out_sf_dict = {
                "input_reals": input_reals[idx : idx + 1],
                "processed_input_reals": processed_input_reals[idx : idx + 1],
                "pre_bottleneck_latents": encoder_infos["pre_bottleneck_latents"][
                    idx : idx + 1
                ],
                "latents": latents[idx : idx + 1],
                "decoded_reals": decoded_reals[idx : idx + 1],
                "trimmed_input_reals": trimmed_input_reals[idx : idx + 1],
                "padding_mask": info["padding_mask"],
            }

            output_subdir = "/".join(info["relpath"].split("/")[:-1])

            import pathlib

            pathlib.Path(f"{self.output_path}/{output_subdir}").mkdir(
                parents=True, exist_ok=True
            )

            fname = info["relpath"].replace(".wav", "").split("/")[-1]

            out_sf_path = f"{self.output_path}/{output_subdir}/{fname}.safetensors"
            out_metadata_path = f"{self.output_path}/{output_subdir}/{fname}.json"

            sf_save_file(out_sf_dict, out_sf_path)

            with open(out_metadata_path, "w") as f:
                json_dump(out_metadata_dict, f, indent=4)

            og_trim_audio = rearrange(
                trimmed_input_reals[idx : idx + 1], "b d n -> d (b n)"
            )
            og_trim_audio = (
                og_trim_audio.to(torch.float32)
                .clamp(-1, 1)
                .mul(32767)
                .to(torch.int16)
                .cpu()
            )

            recon_audio = rearrange(decoded_reals[idx : idx + 1], "b d n -> d (b n)")
            recon_audio = (
                recon_audio.to(torch.float32)
                .clamp(-1, 1)
                .mul(32767)
                .to(torch.int16)
                .cpu()
            )

            torchaudio.save(
                f"{self.output_path}/{output_subdir}/{fname}_original_trimmed.wav",
                og_trim_audio,
                info["sample_rate"],
            )
            torchaudio.save(
                f"{self.output_path}/{output_subdir}/{fname}_reconstructed.wav",
                recon_audio,
                info["sample_rate"],
            )


def load_model(
    model_config=None,
    model_ckpt_path=None,
    pretrained_name=None,
    pretransform_ckpt_path=None,
    model_half=False,
):
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print("Creating model from config")
        model = create_model_from_config(model_config)

        print("Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
        # model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(
            load_ckpt_state_dict(pretransform_ckpt_path), strict=False
        )
        print("Done loading pretransform")

    if model_half:
        model.to(torch.float16)

    print("Done loading model")

    return model, model_config


class ExceptionCallback(Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = get_all_args(
        defaults_file="/home/eduardo/Projects/pre_encode_audio/stable-audio-tools/pre_encode_defaults.ini"
    )
    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    pl.seed_everything(seed, workers=True)

    model, model_config = load_model(
        model_config=args.model_config if args.model_config != "" else None,
        model_ckpt_path=args.pretrained_ckpt_path
        if args.pretrained_ckpt_path != ""
        else None,
        pretrained_name=args.pretrained_name if args.pretrained_name != "" else None,
        pretransform_ckpt_path=args.pretransform_ckpt_path
        if args.pretransform_ckpt_path != ""
        else None,
        model_half=False,
    )

    # Create Dataset
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    pre_encode_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        shuffle=False,
    )

    exc_callback = ExceptionCallback()
    pred_writer = PreEncodePredictionWriter(args.save_dir, write_interval="batch")
    # Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})

    # Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy

            strategy = DeepSpeedStrategy(
                stage=2,
                contiguous_gradients=True,
                overlap_comm=True,
                reduce_scatter=True,
                reduce_bucket_size=5e8,
                allgather_bucket_size=5e8,
                load_full_weights=True,
            )
        else:
            strategy = args.strategy
    else:
        strategy = "ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto"

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        precision=args.precision,
        callbacks=[exc_callback, pred_writer],
        default_root_dir=args.save_dir,
        enable_progress_bar=True,
        strategy=strategy,
        profiler="simple",
        reload_dataloaders_every_n_epochs=0,
    )
    print("Creating pre-encode pipeline")
    pre_encode_pipeline = PreEncodePipeline(
        pretransform=model.pretransform,
        model_config=model_config,
        dataset_config=dataset_config,
    )

    print("Starting prediction")
    trainer.predict(pre_encode_pipeline, pre_encode_dl, return_predictions=False)


if __name__ == "__main__":
    main()
