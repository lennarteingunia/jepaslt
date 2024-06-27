
import importlib
import lightning

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch

from src.models.vjepa import GlossVJEPA
from src.datasets.phoenix14t import Phoenix14T

def cli_main():
    cli = LightningCLI(
        model_class=GlossVJEPA, 
        datamodule_class=Phoenix14T
    )

if __name__ == "__main__":

    cli_main()

    # parser = get_argument_parser()
    # program_args = parser.parse_args()
    # cfg = load_config(program_args.config)
    # main(program_args, cfg)
