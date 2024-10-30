import argparse
import logging
import torch.nn.functional as F

import decord
import matplotlib
import matplotlib.pyplot
import pandas as pd
import torch
import torchvision
import tqdm
from evals.video_classification_frozen.utils import ClipAggregation
from experiments import build_attentive_classifier, build_encoder, load_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run evaluation using V-JEPA.')
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--classifier-checkpoint-path',
                        type=str, required=True)
    parser.add_argument('--overwrite-encoder-checkpoint',
                        required=False, default=None)
    parser.add_argument(
        '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = load_config(config_path=args.config_path)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info('Building encoder and classifier')

    encoder = build_encoder(
        config=config,
        overwrite_checkpoint=args.overwrite_encoder_checkpoint
    ).to(args.device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = ClipAggregation(
        encoder,
        tubelet_size=2,
        attend_across_segments=config.get('optimization').get(
            'attend_across_segments', False)
    )

    classifier = build_attentive_classifier(
        config, checkpoint_path=args.classifier_checkpoint_path).to(args.device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    logger.info(f'Building video loader')

    video_paths = []
    labels = []
    for dataset in config.get('data').get('dataset_val'):
        data = pd.read_csv(dataset, header=None, delimiter=' ')
        video_paths += list(data.values[:, 0])
        labels += list(data.values[:, 1])

    decord.bridge.set_bridge('torch')
    video_loader = decord.VideoLoader(
        video_paths,
        ctx=decord.cpu(0),
        shape=(16, 224, 224, 3),
        interval=0,
        skip=0,
        shuffle=0,
    )

    normalization = torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    for itr, (clips, indices) in tqdm.tqdm(enumerate(video_loader)):
        # I need B, C, T, H, W dimensions

        # This is what originallly happens in the VideoDataset
        clips = clips.permute(0, 3, 1, 2).float() / 255
        clips = normalization(clips)
        clips = clips.permute(1, 0, 2, 3).unsqueeze(0)
        clips = [clip for clip in clips]
        label = indices[:, 0][0]
        label = torch.tensor([labels[label]])
        clip_indices = [indices[:, 1]]

        # This is what happens in the original eval.py

        clips = [[clip.to(args.device).unsqueeze(0) for clip in clips]]
        clip_indices = [d.to(args.device, non_blocking=True)
                        for d in clip_indices]
        video_ids = [indices[:, 0]]
        print(video_ids)
        labels = label.to(args.device)
        batch_size = len(label)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                outputs = [classifier(o) for o in outputs]

                print(outputs)
                print(labels)

                break
