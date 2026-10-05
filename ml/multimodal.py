"""Small multimodal encoders, cross-modal InfoNCE, Grad-CAM, and MC Dropout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
import joblib
from torch import nn
from torch.nn import functional as F


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.25),
                                     nn.Linear(128, embedding_dim), nn.ReLU())

    def forward(self, x):
        return self.network(x.float())


class ECGEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.ReLU(),
                                      nn.MaxPool1d(2), nn.Conv1d(16, 32, 5, stride=2, padding=2),
                                      nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(64, embedding_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return F.relu(self.projection(self.pool(self.features(x.float())).squeeze(-1)))


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                                      nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, stride=2, padding=1),
                                      nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(64, embedding_dim)

    def forward(self, x):
        return F.relu(self.projection(self.pool(self.features(x.float())).flatten(1)))


class MultimodalCVDModel(nn.Module):
    """Concatenation fusion with explicit modality-availability indicators."""
    def __init__(self, clinical_dim: int, embedding_dim: int = 64, classes: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.clinical_encoder = ClinicalEncoder(clinical_dim, embedding_dim)
        self.ecg_encoder = ECGEncoder(embedding_dim)
        self.image_encoder = ImageEncoder(embedding_dim)
        self.fusion = nn.Sequential(nn.Linear(embedding_dim * 3 + 3, 128), nn.ReLU(),
                                    nn.Dropout(0.35), nn.Linear(128, classes))

    def forward(self, clinical=None, ecg=None, image=None):
        candidates = (("clinical", clinical, self.clinical_encoder),
                      ("ecg", ecg, self.ecg_encoder), ("image", image, self.image_encoder))
        batch = next((x.shape[0] for _, x, _ in candidates if x is not None), None)
        if batch is None:
            raise ValueError("At least one modality must be provided")
        device = next(self.parameters()).device
        embeddings = {}
        available = []
        for name, value, encoder in candidates:
            if value is None:
                embeddings[name] = torch.zeros(batch, self.embedding_dim, device=device)
                available.append(torch.zeros(batch, 1, device=device))
            else:
                embeddings[name] = encoder(value)
                available.append(torch.ones(batch, 1, device=device))
        fused = torch.cat([embeddings["clinical"], embeddings["ecg"], embeddings["image"], *available], dim=1)
        return self.fusion(fused), embeddings


class UnimodalCVDClassifier(nn.Module):
    """Single-branch comparison model with its own binary prediction head."""
    def __init__(self, modality: str, clinical_dim: int, embedding_dim: int = 64):
        super().__init__()
        encoders = {"clinical": lambda: ClinicalEncoder(clinical_dim, embedding_dim),
                    "ecg": lambda: ECGEncoder(embedding_dim),
                    "image": lambda: ImageEncoder(embedding_dim)}
        if modality not in encoders:
            raise ValueError(f"Unknown modality: {modality}")
        self.modality = modality
        self.encoder = encoders[modality]()
        self.head = nn.Linear(embedding_dim, 2)

    def forward(self, x):
        return self.head(self.encoder(x))


def cross_modal_infonce(embeddings: Mapping[str, torch.Tensor], patient_ids: list[str], temperature: float = 0.1):
    """Symmetric InfoNCE for correctly ID-aligned clinical/ECG/image positive pairs."""
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Each contrastive batch must contain one example per patient ID")
    present = [embeddings[k] for k in ("clinical", "ecg", "image") if k in embeddings]
    if len(present) < 2:
        raise ValueError("Contrastive learning requires at least two available modalities")
    if len(patient_ids) < 2:
        raise ValueError("InfoNCE batch needs at least two distinct patients for negatives")
    labels = torch.arange(len(patient_ids), device=present[0].device)
    loss_terms = []
    normalized = [F.normalize(value, dim=1) for value in present]
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            logits = normalized[i] @ normalized[j].T / temperature
            loss_terms.extend((F.cross_entropy(logits, labels), F.cross_entropy(logits.T, labels)))
    return torch.stack(loss_terms).mean()


def mc_dropout_predict(model: nn.Module, inputs: dict, passes: int = 30) -> dict:
    if passes < 2:
        raise ValueError("At least two stochastic passes are required")
    was_training = model.training
    model.train()  # keep dropout active for sampling
    draws = []
    try:
        with torch.no_grad():
            for _ in range(passes):
                logits, _ = model(**inputs)
                draws.append(torch.softmax(logits, dim=-1))
    finally:
        model.train(was_training)
    probs = torch.stack(draws)
    mean = probs.mean(dim=0)
    variance = probs.var(dim=0, unbiased=False).mean(dim=-1)
    return {"mean_probability": mean, "variance": variance,
            "std": variance.sqrt(), "passes": passes,
            "method": "Monte Carlo Dropout predictive variance"}


def gradcam_1d(model: MultimodalCVDModel, signal: torch.Tensor, *, clinical=None,
               image=None, target_class: int | None = None) -> torch.Tensor:
    """Return 1D Grad-CAM for a selected (or predicted) model output class."""
    activation, gradient = {}, {}
    layer = model.ecg_encoder.features[-2]
    handle_a = layer.register_forward_hook(lambda _m, _i, out: activation.setdefault("value", out))
    def capture_gradient(_module, _grad_input, grad_output):
        gradient["value"] = grad_output[0]
    handle_g = layer.register_full_backward_hook(capture_gradient)
    try:
        model.zero_grad(set_to_none=True)
        logits, _ = model(clinical=clinical, ecg=signal, image=image)
        target = logits.argmax(dim=1) if target_class is None else torch.full(
            (len(signal),), target_class, device=logits.device, dtype=torch.long)
        logits.gather(1, target[:, None]).sum().backward()
        weights = gradient["value"].mean(dim=-1, keepdim=True)
        cam = F.relu((weights * activation["value"]).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=signal.shape[-1], mode="linear", align_corners=False).squeeze(1)
        return cam / cam.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    finally:
        handle_a.remove(); handle_g.remove()


def gradcam_2d(model: MultimodalCVDModel, image: torch.Tensor, *, clinical=None,
               ecg=None, target_class: int | None = None) -> torch.Tensor:
    """Return 2D Grad-CAM for a selected (or predicted) model output class."""
    activation, gradient = {}, {}
    layer = model.image_encoder.features[-2]
    handle_a = layer.register_forward_hook(lambda _m, _i, out: activation.setdefault("value", out))
    def capture_gradient(_module, _grad_input, grad_output):
        gradient["value"] = grad_output[0]
    handle_g = layer.register_full_backward_hook(capture_gradient)
    try:
        model.zero_grad(set_to_none=True)
        logits, _ = model(clinical=clinical, ecg=ecg, image=image)
        target = logits.argmax(dim=1) if target_class is None else torch.full(
            (len(image),), target_class, device=logits.device, dtype=torch.long)
        logits.gather(1, target[:, None]).sum().backward()
        weights = gradient["value"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation["value"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        return cam / cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    finally:
        handle_a.remove(); handle_g.remove()


def save_multimodal_checkpoint(model: MultimodalCVDModel, directory: str | Path,
                               clinical_preprocessor, metadata: dict):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "multimodal_weights.pt")
    joblib.dump(clinical_preprocessor, directory / "clinical_preprocessor.joblib")
    config = {"clinical_dim": model.clinical_encoder.network[0].in_features,
              "embedding_dim": model.embedding_dim, "num_classes": model.fusion[-1].out_features}
    (directory / "model_config.json").write_text(json.dumps({**config, **metadata}, indent=2))


def load_multimodal_checkpoint(directory: str | Path, *, map_location="cpu"):
    directory = Path(directory)
    config = json.loads((directory / "model_config.json").read_text())
    model = MultimodalCVDModel(config["clinical_dim"], config["embedding_dim"], config["num_classes"])
    state = torch.load(directory / "multimodal_weights.pt", map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    preprocessor = joblib.load(directory / "clinical_preprocessor.joblib")
    return model, preprocessor, config
