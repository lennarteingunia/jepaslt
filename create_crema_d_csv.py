import glob
from math import floor
import os
import random
from typing import Literal
import click
import pandas as pd

from jinja2 import Environment, FileSystemLoader
from sklearn.model_selection import train_test_split


@click.group('cli')
def cli() -> None:
    pass


@cli.command()
@click.argument('root', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--k', default=5)
def run(root, output_dir, k):

    click.echo(f'Reading dataset at {click.format_filename(root)}')
    paths = list(sorted(glob.glob(os.path.join(root, 'VideoFlash', '*.flv'))))
    click.echo(f'Found a total of {len(paths)} videos.')

    def get_info_from_filepath(path: str):
        base_name = os.path.basename(path)
        file_name, _ = os.path.splitext(base_name)
        actor_id, sentence, emotion, intensity = file_name.split('_')
        return dict(
            actor_id=actor_id,
            sentence=sentence,
            emotion=emotion,
            intensity=intensity
        )

    infos = [get_info_from_filepath(path) for path in paths]
    actors = list(set(info['actor_id'] for info in infos))
    emotions = list(set(info['emotion'] for info in infos))

    df = pd.DataFrame([dict(path=path) | info for path,
                      info in zip(paths, infos)])

    template = Environment(loader=FileSystemLoader('./configs/templates'),
                           trim_blocks=True, lstrip_blocks=True).get_template('vith16_384_16x8x3.j2')

    for split_num in range(k):
        train_split = random.sample(actors, floor(0.8 * len(actors)))
        val_split = [actor for actor in actors if actor not in train_split]

        train_split = df[df['actor_id'].isin(train_split)][['path', 'emotion']]
        val_split = df[df['actor_id'].isin(val_split)][['path', 'emotion']]

        train_split_path = get_split_file_path(output_dir, split_num, 'train')
        train_split.to_csv(train_split_path, sep=' ',
                           index=False, header=False)
        click.echo(f'Wrote training split {split_num} to {train_split_path}')

        val_split_path = get_split_file_path(output_dir, split_num, 'val')
        val_split.to_csv(val_split_path, sep=' ', index=False, header=False)
        click.echo(f'Wrote validation split {split_num} to {val_split_path}')

        config = template.render(
            dict(
                tag=f'crema_d_{split_num}',
                train_split_path=train_split_path,
                val_split_path=val_split_path
            )
        )

        config_file_path = get_config_file_path(split_num)
        with open(config_file_path, 'w') as config_file:
            config_file.write(config)
        click.echo(f'Wrote new config file to {config_file_path}')


@cli.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--k', default=5)
def clean(output_dir, k) -> None:
    for split_num in range(k):
        config_file_path = get_config_file_path(split_num=split_num)
        os.remove(config_file_path)
        click.echo(f'Removed config file from {config_file_path}')

        train_split_path = get_split_file_path(
            output_dir=output_dir, split_num=split_num, split='train')
        os.remove(train_split_path)
        click.echo(f'Removed training split from {train_split_path}')

        val_split_path = get_split_file_path(output_dir, split_num, 'val')
        os.remove(val_split_path)
        click.echo(f'Removed validation split from {val_split_path}')


def get_config_file_path(split_num: int) -> str:
    return f'./configs/evals/vith16_384_crema_d_split_{split_num}_16x8x3.yaml'


def get_split_file_path(output_dir: str, split_num: int, split: Literal['train', 'val']) -> str:
    return os.path.join(output_dir, f'{split}_{split_num}.csv')


if __name__ == '__main__':
    cli()
