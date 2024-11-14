import time
from typing import List, Optional, Tuple, Union
import warnings
import torch
import tqdm

from torch.nn.attention import SDPBackend
from models.vision_transformer import VisionTransformer
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg


class ToMeException(Exception):

    def __init__(self, msg: str):
        super(ToMeException, self).__init__(msg)


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1. - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


def make_attention_class(attention_class):
    class ToMeAttention(attention_class):

        def forward(
            self,
            x: torch.Tensor,
            *,
            _mask: Optional[torch.Tensor] = None,
            size: Optional[torch.Tensor] = None
        ):
            B, N, C = x.shape
            q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                          self.num_heads).permute(2, 0, 3, 1, 4)
            if self.use_sdpa:
                with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
                    x = torch.nn.functional.scaled_dot_product_attention(
                        query=q,
                        key=k,
                        value=v,
                        dropout_p=self.proj_drop_prob
                    )
                    attn = None
            else:
                attn = (q @ k.transpose(-2, -1)) * self.scale

                if size is not None:

                    attn = attn + size.log()[:, None, None, :, 0]

                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = (attn @ v)

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn, k.mean(1)

    class ToMeCrossAttention(attention_class):

        def forward(
            self,
            q: torch.Tensor,
            x: torch.Tensor,
            *,
            mask: Optional[torch.Tensor] = None,
            size: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            B, N_1, C = q.shape
            q = self.q(q).reshape(B, N_1, self.num_heads, C //
                                  self.num_heads).permute(0, 2, 1, 3)
            _, N_2, _ = x.shape
            k, v = self.kv(x).reshape(B, N_2, 2, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)
            if self.use_sdpa:
                with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
                    q = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, dropout_p=self.proj_drop_prob)
            else:
                attn = (q @ k.transpose(-2, -1)) * self.scale

                if size is not None:

                    att = attn + size.log()[:, None, None, :, 0]

                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                q = (attn @ v)

            q = q.transpose(1, 2).reshape(B, N_1, C)
            q = self.proj(q)
            q = self.proj_drop(q)
            return q

    if attention_class.__name__ == 'Attention':
        return ToMeAttention
    elif attention_class.__name__ == 'CrossAttention':
        return ToMeCrossAttention
    else:
        raise ToMeException(
            f'Cannot determine ToMe patching of class {attention_class}.')


def make_block_class(block_class):

    class ToMeAttentionBlock(block_class):

        def forward(
            self,
            x: torch.Tensor,
            return_attention: bool = False,
            mask: Optional[torch.Tensor] = None
        ) -> None:

            attn_size = self._tome_info['size'] if self._tome_info['prop_attn'] else None

            y, attn, metric = self.attn(self.norm1(x), size=attn_size)

            if return_attention:
                return attn

            x = x + y

            r = self._tome_info['r'].pop(0)
            if r > 0:

                merge, _ = bipartite_soft_matching(
                    metric=metric,
                    r=r,
                    class_token=self._tome_info['class_token'],
                    distill_token=self._tome_info['distill_token'],
                )

                if self._tome_info['trace_source']:
                    self._tome_info['source'] = merge_source(
                        merge=merge,
                        x=x,
                        source=self._tome_info['source']
                    )

                x, self._tome_info['size'] = merge_wavg(
                    merge=merge,
                    x=x,
                    size=self._tome_info['size']
                )

            return x + self.mlp(self.norm2(x))

    class ToMeCrossAttentionBlock(block_class):

        def forward(
            self,
            q: torch.Tensor,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:

            attn_size = self._tome_info['size'] if self._tome_info['prop_attn'] else None
            y, metric = self.xattn(q, self.norm1(
                x), _mask=mask, size=attn_size)
            q = q + y
            r = self._tome_info['r'].pop(0)
            if r > 0:

                merge, _ = bipartite_soft_matching(
                    metric=metric,
                    r=r,
                    class_token=self._tome_info['class_token'],
                    distill_token=self._tome_info['distill_token'],
                )

                if self._tome_info['trace_source']:
                    self._tome_info['source'] = merge_source(
                        merge=merge,
                        x=x,
                        source=self._tome_info['source']
                    )

                q, self._tome_info['size'] = merge_wavg(
                    merge=merge,
                    x=x,
                    size=self._tome_info['size']
                )

            return q + self.mlp(self.norm2(q))

    if block_class.__name__ == 'Block':
        return ToMeAttentionBlock
    elif block_class.__name__ == 'CrossAttentionBlock':
        return ToMeCrossAttentionBlock
    else:
        raise ToMeException(
            f'Cannot determine ToMe block class of class {block_class}')


def make_vision_transformer_class(transformer_class):

    class ToMeVisionTransformer(transformer_class):

        def forward(self, *args, **kwargs) -> torch.Tensor:
            self._tome_info['r'] = parse_r(len(self.blocks), self.r)
            self._tome_info['size'] = None
            self._tome_info['source'] = None
            return super(ToMeVisionTransformer, self).forward(*args, **kwargs)

    return ToMeVisionTransformer


def apply_patch(model, trace_source: bool = False, prop_attn: bool = False):

    if model.__class__.__name__ == 'ToMeVisionTransformer':
        warnings.warn(
            f'Not patching the given model, since it has already been patched previously.')
        return

    BlockClass = None
    AttentionClass = None
    TransformerClass = model.__class__

    for module in model.modules():
        if module.__class__.__name__ == 'Block':
            BlockClass = module.__class__
        elif module.__class__.__name__ == 'Attention':
            AttentionClass = module.__class__
        elif module.__class__.__name__ == 'CrossAttentionBlock':
            BlockClass = module.__class__
        elif module.__class__.__name__ == 'CrossAttention':
            AttentionClass = module.__class__

    if BlockClass is None or AttentionClass is None:
        warnings.warn(
            f'Error patching model of type {model.__class__.__name__}. It is not a Vision Transformer')

    ToMeAttention = make_attention_class(AttentionClass)
    ToMeBlock = make_block_class(BlockClass)
    ToMeVisionTransformer = make_vision_transformer_class(TransformerClass)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        'r': model.r,
        'size': None,
        'source': None,
        'trace_source': trace_source,
        'prop_attn': prop_attn,
        'class_token': False,
        'distill_token': False,
    }

    for module in model.modules():
        if isinstance(module, BlockClass):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, AttentionClass):
            module.__class__ = ToMeAttention
            module._tome_info = model._tome_info

    return model


def benchmark(
    model: VisionTransformer,
    device: torch.device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
    input_size: Tuple[int] = (3, 244, 244),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_bfloat16: bool = True,
    verbose: bool = True
) -> float:
    model = model.eval().to(device)
    input = torch.rand(batch_size, *input_size, device=device)
    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()

    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_bfloat16):
        with torch.no_grad():
            for itr in tqdm.tqdm(range(runs), disable=not verbose, desc='Benchmarking'):
                if itr == warm_up:
                    torch.cuda.synchronize()
                    total = 0
                    start = time.time()
                model(input)
                total += batch_size

    torch.cuda.synchronize()
    end = time.time()
    elapsed = end - start
    throughput = total / elapsed

    if verbose:
        print(f'Throughput: {throughput:.2f} clips/s')

    return throughput


if __name__ == '__main__':
    import argparse
    import yaml
    import os
    from evals.pretrained_full_video_classification.eval import init_model

    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', required=True)
    parser.add_argument('--local-checkpoint', default=None)
    args = parser.parse_args()

    with open(args.fname, 'r') as f:
        args_eval = yaml.load(f, Loader=yaml.FullLoader)

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    if args.local_checkpoint is not None:
        pretrained_path = args.local_checkpoint
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    val_data_path = args_data.get('dataset_val')
    if not (type(val_data_path) is list):
        val_data_path = [val_data_path]
    num_classes = args_data.get('num_classes')
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)
    use_tome = args_opt.get('use_tome', False)

    eval_tag = args_eval.get('tag', None)

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa
    )

    before_tp = benchmark(encoder, device=device, input_size=(
        3, 16, 384, 384), batch_size=2, use_bfloat16=True)

    encoder = apply_patch(encoder, trace_source=False, prop_attn=False)
    encoder.r = 45

    after_tp = benchmark(encoder, device=device, input_size=(
        3, 16, 384, 384), batch_size=2, use_bfloat16=True)

    print(f"There was a x{after_tp / before_tp:.2f} throughput speedup.")
