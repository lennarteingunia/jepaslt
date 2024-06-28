from lightning.pytorch.cli import LightningCLI

from lightning_modules.jepaslt import JepaSLTStage1
from lightning_data_modules.phoenix14t import Phoenix14T

def cli_main():
    cli = LightningCLI(
        model_class=JepaSLTStage1, 
        datamodule_class=Phoenix14T
    )

if __name__ == "__main__":

    cli_main()

    # parser = get_argument_parser()
    # program_args = parser.parse_args()
    # cfg = load_config(program_args.config)
    # main(program_args, cfg)
