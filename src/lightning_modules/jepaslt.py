import logging
import sys
import loralib
import lightning
import torch
from evals.video_classification_frozen.utils import ClipAggregation
from models.lora_vision_transformer import LoRAVisionTransformer
import src.models.vision_transformer as vision_transformer
from utils.schedulers import CosineWDSchedule, LRWDSchedule, WarmupCosineLRSchedule

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

        self.automatic_optimization = False  # IMPORTANT!

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

        self.warmup = 2  # TODO: This needs to be included in the training parameters

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

        logger.info(
            f'Loading pretrained encoder model from {self.checkpoint_path}')
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        pretrained_dict = checkpoint[self.checkpoint_key]
        pretrained_dict = {
            k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        pretrained_dict = {
            k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}

        for k, v in encoder.state_dict().items():
            if k not in pretrained_dict:
                logger.info(
                    f"Key '{k}' could not be found in loaded state dict")
            elif pretrained_dict[k].shape != v.shape:
                logger.info(
                    f"Key '{k}' is of different shape in model and loaded state dict")
                pretrained_dict[k] = v

        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"Loaded pretrained encoder model with msg: {msg}")
        logger.info(
            f"Loaded pretrained encoder model from epoch: {checkpoint['epoch']}")
        del checkpoint

        # TODO: Rank parameters and the like.
        encoder = LoRAVisionTransformer.from_vit(encoder)
        # This way only the LoRA are trainable
        loralib.mark_only_lora_as_trainable(encoder)

        # if pretrain_frames_per_clip == 1:
        #     # Process each frame independently and aggregate
        #     encoder = FrameAggregation(encoder).to(device)
        # We might to do it like this in the future in case I want to process single images as well.
        self.encoder = ClipAggregation(
            encoder,
            tubelet_size=self.tubelet_size,
            # attend_across_segments=attend_across_segments This will cause the clip aggregation to also attend across different video segments. I don't think I need this.
        )

        # TODO: Load the predictor as well.

    def training_step(self, batch, batch_idx):

        clips, clip_indices, label = batch['clips'], batch['clip_indices'], batch['label']
        clips = [[dji for dji in di] for di in clips]
        clip_indices = [d for d in clip_indices]

        outputs = self.encoder(clips, clip_indices)
        # TODO: wrap in if statement
        if True: #  This is supposed to be 'attend_across_segments'
            outputs = []

        # TODO: Configure gradient clipping
        optimizer = self.optimizers()
        optimizer.step()
        scheduler = self.lr_schedulers()
        scheduler.step()  # This was moved to the bottom for interface conformity

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
        optimizer = torch.optim.AdamW(param_groups)
        total_iterations = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_iterations * self.warmup /
                           self.trainer.max_epochs)
        # TODO: These values are still needed
        lr_scheduler = WarmupCosineLRSchedule(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            start_lr=0,
            ref_lr=0,
            final_lr=0,
            T_max=total_iterations
        )
        wd_scheduler = CosineWDSchedule(
            optimizer=optimizer,
            ref_wd=0,
            final_wd=0,
            T_max=total_iterations
        )
        # Luckily there are no type checks being performed here. This means we can fake a learning rate scheduler, that also schedules weight decay.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LRWDSchedule(
                    lr_scheduler=lr_scheduler,
                    wd_scheduler=wd_scheduler
                ),
                "frequency": 1
            }
        }
