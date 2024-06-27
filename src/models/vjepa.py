import logging
import sys
import lightning
import torch
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
import src.models.vision_transformer as vision_transformer
import src.models.predictor as vit_predictor
from utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

class GlossVJEPA(lightning.LightningModule):

    def __init__(
        self,
        device,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        model_name='vit_base',
        crop_size=224,
        pred_depth=6,
        pred_embed_dim=384,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_sdpa=False,
    ) -> None:
        super(GlossVJEPA, self).__init__()

        encoder = vision_transformer.__dict__[model_name](
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
        )
        self.encoder = MultiMaskWrapper(encoder)
        predictor = vit_predictor.__dict__['vit_predictor'](
            img_size=crop_size,
            use_mask_tokens=use_mask_tokens,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=self.encoder.backbone.embed_dim,
            predictor_embed_dim=pred_embed_dim,
            depth=pred_depth,
            num_heads=self.encoder.backbone.num_heads,
            uniform_power=uniform_power,
            num_mask_tokens=num_mask_tokens,
            zero_init_mask_tokens=zero_init_mask_tokens,
            use_sdpa=use_sdpa,
        )
        self.predictor = PredictorMultiMaskWrapper(predictor)

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)

        for m in self.encoder.modules():
            init_weights(m)

        for m in self.predictor.modules():
            init_weights(m)

        self.encoder.to(device)
        self.predictor.to(device)
        logger.info(encoder)
        logger.info(predictor)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f'Encoder number of parameters: {count_parameters(self.encoder)}')
        logger.info(f'Predictor number of parameters: {count_parameters(self.predictor)}')

    def forward(self, x):
        raise NotImplementedError()
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError()
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()
    
    def configure_optimizers(self):
        raise NotImplementedError()