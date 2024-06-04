
import json
import os
import pathlib
import click
import pandas as pd
import tqdm
import definitions
import ffmpeg


def create_vjepa_compatible_WLASL_csv(
    root_dir: str,
    output_dir: str,
) -> None:
    metadata_json_path = os.path.join(
        root_dir, "metadata", f"WLASL_v0.3.json")
    with open(metadata_json_path, "r") as f:
        metadata = json.load(f)

    outputs = {
        "all": []
    }

    glosses = {}

    for gloss_entry in tqdm.tqdm(metadata):

        gloss = gloss_entry["gloss"]
        if gloss in glosses:
            gloss = glosses[gloss]
        else:
            glosses[gloss] = len(glosses)
            gloss = len(glosses) - 1
        

        for instance in tqdm.tqdm(gloss_entry["instances"], leave=False):
            split = instance["split"]
            video_id = instance["video_id"]
            start_frame = int(instance["frame_start"])
            end_frame = int(instance["frame_end"])
            input_path = os.path.join(
                root_dir, "videos", f"{split}", f"{video_id}.mp4")
            if not os.path.exists(os.path.join(output_dir, split)):
                pathlib.Path(os.path.join(output_dir, split)).mkdir(
                    parents=True, exist_ok=True)

            # calculate how many frames there actually are if i want to trim
            video_probe = ffmpeg.probe(input_path)
            video_probe_stream = next(
                (stream for stream in video_probe["streams"] if stream["codec_type"] == "video"), None)
            number_frames = video_probe_stream["nb_frames"]
            end_frame = number_frames if end_frame == -1 else end_frame

            output_path = os.path.join(
                output_dir, f"{split}", f"{video_id}_{start_frame}_{end_frame}.mp4")

            (ffmpeg
                .input(input_path)
                .trim(start_frame=start_frame, end_frame=end_frame)
                .output(output_path, loglevel="quiet")
                .overwrite_output()
                .run_async()
            )

            if split not in outputs:
                outputs[split] = []

            outputs[split].append(
                dict(
                    absolute_path=output_path,
                    label=gloss
                )
            )

            outputs["all"].append(
                dict(
                    absolute_path=output_path,
                    label=gloss
                )
            )

    for split, output in outputs.items():
        df = pd.DataFrame(output)
        df.to_csv(os.path.join(output_dir, f"{split}.csv"), sep=" ", index=False, header=False)
        


_PREPROCESSING_METHODS = {
    "WLASL": create_vjepa_compatible_WLASL_csv
}


@click.command(
    name="preprocess",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True
    )
)
@click.pass_context
@click.option(
    "--dataset-type",
    required=True,
    type=click.Choice(
        list(_PREPROCESSING_METHODS.keys()),
        case_sensitive=False
    ),
    help="The name of the Sign Language dataset you want to preprocess."
)
@click.option(
    "--root-dir",
    required=True,
    help="The root path of the dataset you are going to process."
)
@click.option(
    "--output-dir",
    help="The location where the output file will created. When not given will create the file in the dataset root as 'preprocessed.csv'."
)
@click.option(
    "--relative-paths",
    default=False,
    help="When True will make all paths relative to the project location."
)
def preprocess(
    ctxt: click.core.Context,
    dataset_type: str,
    root_dir: str,
    output_dir: str,
    *,
    relative_paths: bool = False
) -> None:
    if not output_dir:
        output_dir = os.path.join(root_dir, "preprocessed.csv")
    if relative_paths:
        root_dir = os.path.join(definitions.ROOT_DIR, root_dir)
        output_dir = os.path.join(definitions.ROOT_DIR, root_dir)
    processing_methods = {
        k.lower(): v for k, v in _PREPROCESSING_METHODS.items()}

    processing_methods[dataset_type.lower()](root_dir, output_dir)


if __name__ == "__main__":
    preprocess()
