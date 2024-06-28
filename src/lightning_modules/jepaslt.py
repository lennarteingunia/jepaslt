import logging
import sys
import lightning
import torch
from evals.video_classification_frozen.utils import ClipAggregation
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
import src.models.vision_transformer as vision_transformer
import src.models.predictor as vit_predictor
from utils.schedulers import CosineWDSchedule, WarmupCosineSchedule
from utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

class JepaSLTStage1(lightning.LightningModule):

    def __init__(
        self,
        checkpoint_path,
        model_name,
        patch_size: int = 16,
        crop_size: int = 224,
        frames_per_clip: int = 16,
        tubelet_size: int = 2,
        use_sdpa: bool = False,
        use_SiLU: bool = False,
        tight_SiLU: bool = True,
        uniform_power: bool = False,
        checkpoint_key: str = "target_encoder"
    ) -> None:
        super(JepaSLTStage1, self).__init__()

        self.automatic_optimization = False # IMPORTANT!

        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.frames_per_clip = frames_per_clip
        self.tubelet_size = tubelet_size
        self.use_sdpa = use_sdpa
        self.use_SiLU = use_SiLU
        self.tight_SiLU = tight_SiLU
        self.uniform_power = uniform_power
        self.checkpoint_key = checkpoint_key
        

    def setup(self, stage) -> None:
        encoder = vision_transformer.__dict__[self.model_name](
            img_size=self.crop_size,
            patch_size=self.patch_size,
            num_frames=self.frames_per_clip,
            tubelet_size=self.tubelet_size,
            uniform_power=self.uniform_power,
            use_sdpa=self.use_sdpa,
            use_SiLU=self.use_SiLU,
            tight_SiLU=self.tight_SiLU
        )

        logger.info(f'Loading pretrained encoder model from {self.checkpoint_path}')
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        pretrained_dict = checkpoint[self.checkpoint_key]
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}

        for k, v in encoder.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f"Key '{k}' could not be found in loaded state dict")
            elif pretrained_dict[k].shape != v.shape:
                logger.info(
                    f"Key '{k}' is of different shape in model and loaded state dict")
                pretrained_dict[k] = v

        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"Loaded pretrained encoder model with msg: {msg}")
        logger.info(f"Loaded pretrained encoder model from epoch: {checkpoint['epoch']}")
        del checkpoint


        # if pretrain_frames_per_clip == 1:
        #     # Process each frame independently and aggregate
        #     encoder = FrameAggregation(encoder).to(device)
        # We might to do it like this in the future in case I want to process single images as well.
        self.encoder = ClipAggregation(
            encoder,
            tubelet_size=self.tubelet_size,
            # attend_across_segments=attend_across_segments This will cause the clip aggregation to also attend across different video segments. I don't think I need this.
        )

        # TODO: This will also need to be LoRA wrapped.
        # TODO: Load the predictor as well.

    def forward(self, x):
        return 0
    
    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        lr_schedulers = self.lr_schedulers()
        
    
    def validation_step(self, batch, batch_idx):
        pass

    def lr_scheduler_step(self, scheduler):
        scheduler.step()
    
    def configure_optimizers(self):
        param_groups = [
            {
                'params': (p for n, p in self.encoder.named_parameters() if ('bias' not in n) and (len(p.shape) != 1))
            },
            {
                'params': (p for n, p in self.encoder.named_parameters() if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]
        optimizer =  torch.optim.AdamW(param_groups)
        return optimizer