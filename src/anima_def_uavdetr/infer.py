"""Inference surface for DEF-UAVDETR."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from .checkpoints import load_checkpoint
from .config import UavDetrSettings
from .model import DefUavDetr
from .postprocess import postprocess_queries


class DefUavDetrPredictor:
    """Wrap model loading, preprocessing, and detection postprocessing."""

    def __init__(
        self,
        model: DefUavDetr | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        settings: UavDetrSettings | None = None,
    ) -> None:
        self.settings = settings or UavDetrSettings()
        self.device = torch.device(device)
        self.model = model or DefUavDetr()
        self.model.to(self.device)
        if checkpoint_path is not None:
            load_checkpoint(self.model, checkpoint_path, map_location=self.device)
        self.model.eval()

    def _preprocess_array(self, image: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(image, (self.settings.image_size, self.settings.image_size))
        if resized.ndim == 2:
            resized = np.stack([resized, resized, resized], axis=-1)
        if resized.shape[2] == 4:
            resized = resized[:, :, :3]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return tensor

    def _load_path(self, path: str | Path) -> torch.Tensor:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"unable to read image: {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._preprocess_array(rgb)

    def _prepare_images(
        self,
        images: torch.Tensor | np.ndarray | str | Path | Iterable[torch.Tensor | np.ndarray | str | Path],
    ) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            tensor = images.float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            return tensor.to(self.device)

        if isinstance(images, np.ndarray):
            return self._preprocess_array(images).unsqueeze(0).to(self.device)

        if isinstance(images, (str, Path)):
            return self._load_path(images).unsqueeze(0).to(self.device)

        batch = []
        for item in images:
            if isinstance(item, torch.Tensor):
                tensor = item.float()
                batch.append(tensor if tensor.ndim == 3 else tensor.squeeze(0))
            elif isinstance(item, np.ndarray):
                batch.append(self._preprocess_array(item))
            else:
                batch.append(self._load_path(item))
        return torch.stack(batch).to(self.device)

    @torch.inference_mode()
    def predict(
        self,
        images: torch.Tensor | np.ndarray | str | Path | Iterable[torch.Tensor | np.ndarray | str | Path],
        *,
        conf: float = 0.25,
        max_det: int = 300,
    ) -> list[torch.Tensor]:
        batch = self._prepare_images(images)
        pred_boxes, pred_logits = self.model(batch)
        return postprocess_queries(pred_boxes, pred_logits, conf=conf, max_det=max_det)

    @torch.inference_mode()
    def predict_from_path(
        self,
        source: str | Path,
        *,
        conf: float = 0.25,
        max_det: int = 300,
    ) -> dict[Path, torch.Tensor]:
        source = Path(source)
        if source.is_file():
            return {source: self.predict(source, conf=conf, max_det=max_det)[0]}

        supported = {".jpg", ".jpeg", ".png", ".bmp"}
        outputs: dict[Path, torch.Tensor] = {}
        for path in sorted(source.iterdir()):
            if path.suffix.lower() in supported and path.is_file():
                outputs[path] = self.predict(path, conf=conf, max_det=max_det)[0]
        return outputs
