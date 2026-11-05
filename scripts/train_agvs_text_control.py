"""Train the paired AGVS text-input control on the frozen MI training subset."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from modules.dataset import MultimodalNerDataset
from modules.internal_split import sha256_file
from scripts import evaluate_bio_joint_recheck as recheck
from scripts import train_bio_joint_recheck as train

OUTPUT = train.ROOT / 'agvs_text_control'
MASKED_MULTIMODAL_DEV_F1 = 0.8016112789526687


class TextInputDataset(MultimodalNerDataset):
    """Preserve the original text and label pipeline with an empty image input."""

    def _encode_images(self, _image_names):
        return super()._encode_images([])


def main():
    if OUTPUT.exists():
        raise FileExistsError(f'Preserve the existing text control run: {OUTPUT}')
    runtime = recheck.baseline.check_baseline_runtime()
    import torch
    from torch.utils.data import DataLoader, Subset

    from models.agvs_mner import AgvsMnerModel
    from modules.context_consistency import ContextConsistencyTrainer, ContextViewDataset
    from run import configure_logging, set_seed

    prepared, full = train.load_prepared(train.split.DATA)
    rows, positions, binding = train.training_plan(prepared, full)
    seal = recheck.baseline_seal(binding)
    if runtime != seal['runtime'] or seal['best_epoch'] != 11:
        raise ValueError('Expected the completed epoch-11 multimodal control and its original runtime')
    config = train.READ_JSON(seal['config'])
    reference = train.baseline_config(prepared)
    if config != {**reference, 'output_dir': str(train.ROOT / 'baseline')}:
        raise ValueError('Expected the archived multimodal training configuration')
    args = SimpleNamespace(**{**config, 'output_dir': str(OUTPUT)})
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('The text control requires CUDA in Conda base')
    for name in ('bert_model', 'vit_model'):
        if not Path(getattr(args, name)).is_dir():
            raise FileNotFoundError(f'Original local model directory is missing: {name}')

    set_seed(args.seed)
    base = TextInputDataset(args, 'MNER-MI_train.txt')
    context = ContextViewDataset(base, args.internal_manifest, args.augmentation_cache, args.dataset)
    dev = TextInputDataset(args, 'MNER-MI_val.txt')
    if ([base.examples['words'][i] for i in positions] != [row['tokens'] for row in rows['train']]
            or dev.examples['words'] != [row['tokens'] for row in rows['dev']]):
        raise ValueError('Text control inputs differ from the paired training and selection rows')
    fit = Subset(context, positions)
    for sample in (fit[0], dev[0]):
        if sample[3].count_nonzero() or sample[5].count_nonzero():
            raise ValueError('Expected zero image tensors and zero image masks in both data loaders')
    loaders = {'batch_size': args.batch_size, 'num_workers': args.num_workers}
    fit_loader = DataLoader(fit, **loaders, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(dev, **loaders, shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed + 1))
    inputs = {**seal['input_sha256'], str(recheck.BASELINE_REPORT): sha256_file(recheck.BASELINE_REPORT),
              str(train.split.OUTPUT): train.SPLIT_SHA256,
              str(Path(__file__).resolve()): sha256_file(__file__)}

    OUTPUT.mkdir(exist_ok=False)
    configure_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    logger.info('AGVS text-input control with %d training and %d selection rows', len(fit), len(dev))
    labels = base.processor.get_label_mapping(include_subword_label=True)
    model = AgvsMnerModel(list(labels), args)
    trainer = ContextConsistencyTrainer(fit_loader, dev_loader, None, model, labels, args, logger)
    trainer.train()
    recheck.verify_files(inputs)

    text_f1 = float(trainer.best_dev_f1)
    multimodal_f1 = float(seal['best_dev_f1'])
    result = {
        'scope': 'shared original internal selection dev, native AGVS scoring',
        'train_samples': len(fit), 'selection_samples': len(dev),
        'text_best_epoch': trainer.best_dev_epoch, 'multimodal_best_epoch': seal['best_epoch'],
        'text_trained_dev_f1': text_f1, 'multimodal_trained_dev_f1': multimodal_f1,
        'multimodal_checkpoint_masked_dev_f1': MASKED_MULTIMODAL_DEV_F1,
        'multimodal_minus_text_trained_f1': multimodal_f1 - text_f1,
    }
    report_path = OUTPUT / 'completed.json'
    train.WRITE_JSON(report_path, {
        'schema_version': 'agvs-text-control-dev-v1', **result, 'recheck': binding,
        'seed': args.seed, 'image_policy': 'zero pixels and masks in training and selection',
        'runtime': runtime, 'input_sha256': inputs,
        'checkpoint': str(trainer.checkpoint_path.resolve()),
        'output_sha256': {str(path.resolve()): sha256_file(path) for path in (
            trainer.checkpoint_path, trainer.run_directory / 'config.json', trainer.results_path)},
        'grouped_holdout_scored': False, 'official_val_or_test_read': False,
    })
    print(json.dumps({**result, 'report': str(report_path.resolve())}, indent=2), flush=True)


if __name__ == '__main__':
    main()
