import lightning



class WLASL(lightning.LightningDataModule):

    def __init__(
        self,
        root: str,
        *,
        prepare: bool = True
    ) -> None:
        super(WLASL, self).__init__()
        self.root = root
        self.prepare = prepare

    def prepare_data(self) -> None:
        if self.prepare:
            print("Preparing data...")

if __name__ == "__main__":
    dm = WLASL()