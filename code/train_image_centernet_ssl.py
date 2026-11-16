"""Train image centernet with an SSL ResNet-18 backbone, then evaluate on the test split.

[Unified] SSL Wave 1a. Identical to train_image_centernet.py (same protocol, data, augmentation, PreciseBN,
30 epochs, Adam 1e-4, effective batch 8, seed 777) except that after the common initialisation the backbone
(net.conv1 / bn1 / layer1-4, incl. BN statistics) is replaced by an SSL checkpoint. The deformable
up-convolutions and the heads keep the common / official initialisation. No other file of the repository changes.

    python train_image_centernet_ssl.py --pretrained C:\\SSL_Wave1\\runs\\ssl_simclr\\backbone_final.pth

The run folder is named after the SSL run: .../ssl_simclr/backbone_final.pth -> runs/image_centernet_ssl_simclr/.
Copy this file into the CenterNet repository folder (next to train_image_centernet.py).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main                                   # noqa: E402
from detectors.centernet.build import CenterNet                  # noqa: E402

BACKBONE = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")


class CenterNetSSL(CenterNet):
    def load_darknet_weights(self, path):            # called by the runner right after weights_init_normal
        sd = torch.load(path, map_location="cpu")
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        not_loaded = [k for k in missing if k.split(".")[0] in BACKBONE]
        if not_loaded or unexpected:
            raise RuntimeError(f"SSL backbone mismatch: not loaded {not_loaded[:5]}, unexpected {unexpected[:5]}")
        n = sum(v.numel() for k, v in sd.items() if not k.endswith("num_batches_tracked"))
        print(f"SSL backbone : {path}  ({len(sd)} tensors, {n:,} values -> net.*; "
              f"{len(missing)} tensors keep their init: deconv + heads)")


def build_model(task, num_classes):
    return CenterNetSSL(num_classes)


def version_from_args():
    if "--pretrained" not in sys.argv:
        raise SystemExit("--pretrained <backbone.pth> is required (use train_image_centernet.py for scratch)")
    p = Path(sys.argv[sys.argv.index("--pretrained") + 1])
    tag = p.parent.name.replace("ssl_", "", 1) if p.parent.name.startswith("ssl_") else p.stem
    if p.stem != "backbone_final":
        tag += "_" + p.stem.replace("backbone_", "")
    return f"centernet_ssl_{tag}"


if __name__ == "__main__":
    main("train", task="image", version=version_from_args(), build_model=build_model)
