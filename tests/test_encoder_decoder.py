"""Unit tests for the qualia-aware encoder/decoder."""

from __future__ import annotations

import math

import pytest
import torch
from qualia.model.decoder import QualiaDecoder
from qualia.model.encoder import PAYLOAD_KEYS, EncoderOutput, QualiaEncoder

IMG_SIZE = 32
SENSORY_DIM = 32
BATCH = 8


def _make_encoder_decoder(
    seed: int = 0,
    backbone: str = "cnn",
    payload_vocabs: dict[str, int] | None = None,
) -> tuple[QualiaEncoder, QualiaDecoder]:
    torch.manual_seed(seed)
    encoder = QualiaEncoder(
        modality="image",
        backbone=backbone,
        in_channels=3,
        image_size=IMG_SIZE,
        sensory_dim=SENSORY_DIM,
        payload_keys=PAYLOAD_KEYS,
        payload_vocabs=payload_vocabs or {"shape": 4, "color": 4},
    )
    decoder = QualiaDecoder(encoder=encoder)
    return encoder, decoder


def _make_batch(batch: int = BATCH, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(batch, 3, IMG_SIZE, IMG_SIZE) * 2 - 1


# ---------------------------------------------------------------------------
# Smoke / shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone", ["cnn", "vit"])
def test_encoder_produces_structured_outputs(backbone: str) -> None:
    encoder, _ = _make_encoder_decoder(seed=1, backbone=backbone)
    x = _make_batch(seed=2)
    out = encoder(x)

    assert isinstance(out, EncoderOutput)
    assert out.sensory.shape == (BATCH, SENSORY_DIM)
    assert out.backbone_features.shape[0] == BATCH
    assert set(out.payload.keys()) == set(PAYLOAD_KEYS)

    # shape/color categorical; hue/brightness/agency continuous
    assert out.payload["shape"].shape == (BATCH, 4)
    assert out.payload["color"].shape == (BATCH, 4)
    for key in ("hue", "brightness", "agency"):
        assert out.payload[key].shape == (BATCH, 1)

    vec = out.payload_vector()
    expected = sum(encoder.payload_vocabs[k] for k in PAYLOAD_KEYS)
    assert vec.shape == (BATCH, expected)


@pytest.mark.parametrize("backbone", ["cnn", "vit"])
def test_decoder_reconstructs_correct_shape(backbone: str) -> None:
    encoder, decoder = _make_encoder_decoder(seed=3, backbone=backbone)
    x = _make_batch(seed=4)
    out = encoder(x)
    recon = decoder.reconstruct(out)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_decoder_sensory_only_fallback_produces_output() -> None:
    encoder, decoder = _make_encoder_decoder(seed=5)
    x = _make_batch(seed=6)
    out = encoder(x)
    recon = decoder.reconstruct(out, use_payload=False)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_audio_pipeline_runs() -> None:
    torch.manual_seed(0)
    encoder = QualiaEncoder(
        modality="audio",
        in_channels=1,
        sensory_dim=SENSORY_DIM,
        payload_keys=PAYLOAD_KEYS,
        payload_vocabs={"shape": 4, "color": 4},
    )
    decoder = QualiaDecoder(encoder=encoder)
    audio = torch.randn(2, 1, 1024)
    out = encoder(audio)
    recon = decoder.reconstruct(out)
    assert recon.dim() == 3
    assert recon.shape[0] == 2
    assert recon.shape[-1] == audio.shape[-1], (
        f"audio reconstruction length must match input length; "
        f"got {recon.shape[-1]} vs {audio.shape[-1]}"
    )
    assert torch.isfinite(recon).all()


def test_payload_size_matches_vocab_sum() -> None:
    encoder, _ = _make_encoder_decoder(
        seed=7,
        payload_vocabs={"shape": 5, "color": 3},
    )
    expected = 5 + 3 + 1 + 1 + 1
    assert encoder.payload_size() == expected


# ---------------------------------------------------------------------------
# Differentiable + recon loss is finite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone", ["cnn", "vit"])
def test_reconstruction_loss_is_finite(backbone: str) -> None:
    encoder, decoder = _make_encoder_decoder(seed=8, backbone=backbone)
    x = _make_batch(seed=9)
    out = encoder(x)
    recon = decoder.reconstruct(out)
    recon_no_payload = decoder.reconstruct(out, use_payload=False)
    loss = (recon - x).pow(2).mean() + (recon_no_payload - x).pow(2).mean()
    assert torch.isfinite(loss)
    loss.backward()

    params_with_grad = [
        name for name, p in encoder.named_parameters() if p.requires_grad and p.grad is not None
    ]
    assert params_with_grad, "expected gradients on encoder"
    dec_params_with_grad = [
        name for name, p in decoder.named_parameters() if p.requires_grad and p.grad is not None
    ]
    assert dec_params_with_grad, "expected gradients on decoder"


# ---------------------------------------------------------------------------
# Mutual-information-style sanity check
# ---------------------------------------------------------------------------


def _gaussian_mi_lower_bound(x: torch.Tensor, y: torch.Tensor) -> float:
    """Estimate a lower bound on mutual information between two sample batches.

    Uses the ``InfoNCE``-style bound from
    van den Oord et al. (2018). For each row ``i`` the true positive pair is
    ``(x[i], y[i])``; the negatives are ``y[j] for j != i`` (within-batch
    negatives). MI is bounded below by

        ``I(X; Y) >= log(N) - (1/N) * sum_i CE(logits_i, i)``

    i.e. ``log(N) - cross_entropy_loss``, which is what this function returns
    in nats.

    This is a positive-valued statistic whenever ``y`` is correlated with
    ``x`` across the batch (i.e. carries information); for an independent
    noise sample it will be near zero.

    ``x`` and ``y`` need only share the leading ``batch`` dimension; trailing
    dimensions are flattened for the cosine-similarity critic.
    """
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("x and y must share the leading batch dim, batch >= 2")
    n = x.shape[0]
    x_n = torch.nn.functional.normalize(x.reshape(n, -1), dim=-1)
    y_n = torch.nn.functional.normalize(y.reshape(n, -1), dim=-1)
    logits = x_n @ y_n.t()  # (n, n)
    targets = torch.arange(n, device=x.device)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    mi_lower = math.log(n) - float(loss.item())
    return mi_lower


def test_payload_carries_mutual_information_with_input() -> None:
    """The payload must carry non-trivial info about the input.

    Concretely: a small linear probe trained to recover the input image from
    the payload should outperform a probe trained from random Gaussian noise
    of the same shape. We measure this with held-out MSE: lower is better,
    so we assert that the payload-probe MSE is strictly lower than the
    random-probe MSE.

    This is a deterministic, fast proxy for the more expensive MI estimator
    yet still rules out the "payload is decorative" failure mode.
    """
    torch.manual_seed(0)
    encoder, _decoder = _make_encoder_decoder(
        seed=10,
        backbone="cnn",
        payload_vocabs={"shape": 4, "color": 4},
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    n_train = 32
    n_probe = 64
    images = torch.rand(n_train, 3, IMG_SIZE, IMG_SIZE) * 2 - 1
    test_images = torch.rand(n_probe, 3, IMG_SIZE, IMG_SIZE) * 2 - 1

    with torch.no_grad():
        train_payload = encoder(images).payload_vector()
        test_payload = encoder(test_images).payload_vector()
        train_target = images.flatten(1)
        test_target = test_images.flatten(1)

    payload_dim = train_payload.shape[-1]
    probe = torch.nn.Linear(payload_dim, train_target.shape[-1])

    optim_probe = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(40):
        optim_probe.zero_grad()
        pred = probe(train_payload)
        loss = (pred - train_target).pow(2).mean()
        loss.backward()
        optim_probe.step()
    with torch.no_grad():
        test_mse_payload = (probe(test_payload) - test_target).pow(2).mean().item()

    rng = torch.Generator().manual_seed(42)
    random_train = torch.randn(train_payload.shape, generator=rng)
    random_test = torch.randn(test_payload.shape, generator=rng)
    probe_rand = torch.nn.Linear(payload_dim, train_target.shape[-1])
    optim_rand = torch.optim.Adam(probe_rand.parameters(), lr=1e-2)
    for _ in range(40):
        optim_rand.zero_grad()
        pred = probe_rand(random_train)
        loss = (pred - train_target).pow(2).mean()
        loss.backward()
        optim_rand.step()
    with torch.no_grad():
        test_mse_random = (probe_rand(random_test) - test_target).pow(2).mean().item()

    assert test_mse_payload < test_mse_random, (
        f"payload-probe MSE ({test_mse_payload:.4f}) should beat "
        f"random-probe MSE ({test_mse_random:.4f}); the payload is not informative"
    )

    # And a positive InfoNCE-style MI lower bound on (image, payload).
    # Project the payload to the same dim as the image embedding so the
    # cosine-similarity critic can be applied.
    payload_dim = train_payload.shape[-1]
    proj_payload = torch.nn.Linear(payload_dim, 32).eval()
    proj_image = torch.nn.Linear(test_target.shape[-1], 32).eval()
    with torch.no_grad():
        # Use the test images and their payload vectors for the MI lower bound.
        test_emb = proj_image(test_target)
        payload_emb = proj_payload(test_payload)
    mi_lower_payload = _gaussian_mi_lower_bound(test_emb, payload_emb)
    rand_emb = torch.randn_like(test_payload)
    mi_lower_random = _gaussian_mi_lower_bound(test_emb, proj_payload(rand_emb))

    assert mi_lower_payload > mi_lower_random, (
        f"payload MI lower-bound ({mi_lower_payload:.4f}) should exceed the "
        f"random-feature lower-bound ({mi_lower_random:.4f}); payload carries "
        f"no measurable info about the input"
    )
