# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Export CausalAnalysisPredictor hierarchical contexts h_ec, h_gc, h_sc per image (for ablations)."""
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg, coerce_yacs_cli_opts
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import all_gather, is_main_process, synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.amp_compat import amp


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _predictor(model):
    m = _unwrap(model)
    return m.roi_heads.relation.predictor


def main():
    parser = argparse.ArgumentParser(description="Save h_ec, h_gc, h_sc tensors per dataset index")
    parser.add_argument("--config-file", required=True, metavar="FILE")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=0)
    parser.add_argument(
        "--output-features-dir",
        default="output/features",
        help="Directory for per-image .pth files (default: output/features)",
    )
    parser.add_argument(
        "opts",
        help="Modify config options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(coerce_yacs_cli_opts(args.opts))
    cfg.defrost()
    cfg.TEST.EXPORT_HIERARCHICAL_FEATURES = True
    cfg.freeze()

    logger = setup_logger("maskrcnn_benchmark", "", get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    use_mixed_precision = cfg.DTYPE == "float16"
    amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    out_root = os.path.abspath(os.path.expanduser(args.output_features_dir))
    if is_main_process():
        mkdir(out_root)

    dataset_names = cfg.DATASETS.TEST
    if cfg.DATASETS.TO_TEST == "train":
        dataset_names = cfg.DATASETS.TRAIN
    elif cfg.DATASETS.TO_TEST == "val":
        dataset_names = cfg.DATASETS.VAL

    loaders = make_data_loader(
        cfg=cfg, mode="test", is_distributed=distributed, dataset_to_test=cfg.DATASETS.TO_TEST
    )

    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    cpu = torch.device("cpu")

    pred_mod = _predictor(model)

    for dataset_name, data_loader in zip(dataset_names, loaders):
        out_dir = os.path.join(out_root, dataset_name.replace("/", "_"))
        if is_main_process():
            mkdir(out_dir)
        if pred_mod.__class__.__name__ != "CausalAnalysisPredictor":
            logger.warning(
                "Predictor is %s, not CausalAnalysisPredictor; h_ec/h_gc/h_sc may be absent.",
                pred_mod.__class__.__name__,
            )

        for _, batch in enumerate(tqdm(data_loader, desc=dataset_name)):
            images, targets, image_ids = batch
            targets = [t.to(device) for t in targets]
            with torch.no_grad():
                _ = model(images.to(device), targets)

            exp = getattr(pred_mod, "_hierarchical_export", None)
            if exp is None:
                logger.warning("No _hierarchical_export; set TEST.EXPORT_HIERARCHICAL_FEATURES True (script does this by default).")
                continue

            h_ec = exp["h_ec"]
            h_gc = exp["h_gc"]
            h_sc = exp["h_sc"]
            ds = data_loader.dataset

            payloads = []
            for j, idx in enumerate(image_ids):
                idx = int(idx)
                vg_id = None
                try:
                    info = ds.get_img_info(idx)
                    vg_id = info.get("image_id", None)
                except Exception:
                    pass
                if vg_id is None:
                    try:
                        base = os.path.basename(ds.filenames[idx])
                        vg_id = int(os.path.splitext(base)[0])
                    except Exception:
                        vg_id = idx

                blob = {
                    "dataset_index": idx,
                    "vg_image_id": vg_id,
                    "h_ec": h_ec[j].detach().to(cpu).clone(),
                    "h_gc": h_gc[j].detach().to(cpu).clone(),
                    "h_sc": h_sc[j].detach().to(cpu).clone(),
                }
                payloads.append((idx, blob))

            if distributed:
                gathered = all_gather(payloads)
                if not is_main_process():
                    continue
                to_write = []
                for part in gathered:
                    to_write.extend(part)
            else:
                to_write = payloads

            for idx, blob in to_write:
                vg_id = blob.get("vg_image_id", idx)
                try:
                    fname = "{}.pth".format(int(vg_id))
                except (TypeError, ValueError):
                    fname = "{:07d}.pth".format(idx)
                path = os.path.join(out_dir, fname)
                torch.save(blob, path)

        synchronize()

    if is_main_process():
        logger.info("Saved hierarchical features under %s", out_root)


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
