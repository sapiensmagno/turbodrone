"""The scene the downward camera looks at.

Optical flow lives entirely on ground texture, so the texture is not decoration:
it decides ``n_features``, ``inlier_ratio`` and therefore ``quality``, which are
the signals the autopilot gates on. A checkerboard would track perfectly and
prove nothing; a blank floor would never track. Both the texture statistics and
the departure from flatness are calibrated against real recordings -- see
``tests/test_sim_realism.py``, which asserts the sim lands inside the recorded
ranges rather than merely looking plausible.

Two design points worth stating outright:

- **Sparse strong speckles, weak background.** A strong broadband background
  makes ``goodFeaturesToTrack`` return its full 250-corner budget of weak,
  self-similar corners. Real sessions report 37-84 features, and those are
  distinctive blobs, not noise maxima.
- **The floor is not a plane.** Cables, rug edges and toys sit above it, and
  their parallax is what produces the outlier tracks behind the recorded
  ``inlier_ratio`` of ~0.89. A perfectly planar scene tracks at 1.00 and
  flatters the estimator, which is the opposite of what a pre-flight check is
  for. The relief layer is a second plane at ``relief_height_m`` above the
  floor, masked to cover ``relief_fraction`` of the area.

The texture is stored single-channel. Every consumer downstream converts to
grayscale anyway (``LucasKanadeDriftEstimator`` and the ORB reference detector
both start with ``cvtColor(..., BGR2GRAY)``), so carrying three identical
channels through the warp would only cost memory and time. The renderer expands
to BGR at the end, because that is what ``Drone.get_frame`` returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from e88_sim.config import GroundParams


@dataclass(frozen=True)
class GroundTexture:
    """A ground plane image with a known metric scale.

    The world origin sits at the centre of the image. World +x is texture +u,
    world +y is texture -v (because image rows grow downward while world y grows
    forward).

    ``relief_mask`` marks where raised objects sit: 0 = bare floor, 1 = an object
    whose surface is ``relief_height_m`` above the floor. It is ``None`` when the
    scene is perfectly planar.
    """

    gray: np.ndarray  # (H, W) uint8
    m_per_px: float
    relief_mask: Optional[np.ndarray] = None  # (H, W) float32 in [0, 1]
    relief_height_m: float = 0.0

    @property
    def extent_m(self) -> Tuple[float, float]:
        h, w = self.gray.shape[:2]
        return (float(w) * float(self.m_per_px), float(h) * float(self.m_per_px))

    @property
    def center_px(self) -> Tuple[float, float]:
        h, w = self.gray.shape[:2]
        return ((float(w) - 1.0) * 0.5, (float(h) - 1.0) * 0.5)

    @property
    def has_relief(self) -> bool:
        return self.relief_mask is not None and float(self.relief_height_m) > 0.0

    def world_to_texture_px(self, x_m: float, y_m: float) -> Tuple[float, float]:
        u0, v0 = self.center_px
        return (u0 + float(x_m) / float(self.m_per_px), v0 - float(y_m) / float(self.m_per_px))


def _fractal_noise(shape: Tuple[int, int], rng: np.random.Generator, *, octaves: int = 6) -> np.ndarray:
    """Multi-octave value noise, generated small and upsampled per octave.

    Real floors have structure at every scale -- grain, weave, scuffs -- and that
    spread of spatial frequencies is what makes corner detection behave the way
    it does on real footage. White noise alone produces a uniform corner field.
    """
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total = 0.0
    size = 4
    for _ in range(int(octaves)):
        layer = rng.random((max(2, size), max(2, size)), dtype=np.float32)
        layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amplitude * layer
        total += amplitude
        amplitude *= 0.55
        size *= 2
    return out / max(1e-6, total)


def _normalized(a: np.ndarray) -> np.ndarray:
    a = a - float(a.mean())
    peak = float(np.abs(a).max())
    return a / peak if peak > 1e-9 else a


def _blob_field(
    side_px: int,
    rng: np.random.Generator,
    *,
    count: int,
    blob_px: float,
    signed: bool = True,
) -> np.ndarray:
    """Sparse blobs of a given size, as a normalized field in [-1, 1] or [0, 1]."""
    field = np.zeros((side_px, side_px), dtype=np.float32)
    if count <= 0:
        return field
    xs = rng.integers(0, side_px, size=int(count))
    ys = rng.integers(0, side_px, size=int(count))
    if signed:
        # Half bright, half dark, so the mean level is unchanged.
        field[ys, xs] = np.where(rng.random(int(count)) > 0.5, 1.0, -1.0)
    else:
        field[ys, xs] = 1.0
    sigma = max(0.8, float(blob_px) / 3.0)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma)
    peak = float(np.abs(field).max())
    return field / peak if peak > 1e-9 else field


def make_ground(params: Optional[GroundParams] = None) -> GroundTexture:
    """Build the ground texture described by ``params``."""
    p = params or GroundParams()
    rng = np.random.default_rng(int(p.seed))

    mpp = float(p.m_per_px)
    side_px = max(64, int(round(float(p.extent_m) / mpp)))
    area_m2 = float(p.extent_m) ** 2

    img = 128.0 + _normalized(_fractal_noise((side_px, side_px), rng)) * (
        127.0 * float(np.clip(p.background_contrast, 0.0, 1.0))
    )

    n_speckles = int(max(0.0, float(p.speckle_density_per_m2)) * area_m2)
    speckles = _blob_field(side_px, rng, count=n_speckles, blob_px=float(p.speckle_size_m) / mpp)
    img = img + speckles * (127.0 * float(np.clip(p.speckle_contrast, 0.0, 1.0)))

    gray = np.clip(img, 0.0, 255.0).astype(np.uint8)

    relief_mask = _make_relief_mask(side_px, rng, params=p)
    texture = GroundTexture(
        gray=gray,
        m_per_px=mpp,
        relief_mask=relief_mask,
        relief_height_m=float(p.relief_height_m) if relief_mask is not None else 0.0,
    )

    if p.pad_image_path:
        texture = composite_pad(
            texture,
            image_path=str(p.pad_image_path),
            center_m=(float(p.pad_center_m[0]), float(p.pad_center_m[1])),
            size_m=(float(p.pad_size_m[0]), float(p.pad_size_m[1])),
        )
    return texture


def _make_relief_mask(side_px: int, rng: np.random.Generator, *, params: GroundParams) -> Optional[np.ndarray]:
    fraction = float(np.clip(params.relief_fraction, 0.0, 1.0))
    if fraction <= 0.0 or float(params.relief_height_m) <= 0.0:
        return None

    mpp = float(params.m_per_px)
    patch_px = max(2.0, float(params.relief_patch_size_m) / mpp)
    area_m2 = float(params.extent_m) ** 2
    patch_area_m2 = float(params.relief_patch_size_m) ** 2
    count = int(max(1.0, fraction * area_m2 / max(1e-9, patch_area_m2)))

    field = _blob_field(side_px, rng, count=count, blob_px=patch_px, signed=False)
    # Threshold to get crisp object boundaries at roughly the requested coverage;
    # a soft alpha ramp would blend two depths inside one pixel, which is not how
    # an object edge looks and would blunt exactly the parallax we want.
    if field.max() <= 1e-9:
        return None
    threshold = float(np.quantile(field, 1.0 - fraction))
    mask = (field > threshold).astype(np.float32)
    return mask


def composite_pad(
    texture: GroundTexture,
    *,
    image_path: str,
    center_m: Tuple[float, float],
    size_m: Tuple[float, float],
) -> GroundTexture:
    """Paste a physical pad of known size onto the ground.

    This is what makes the visual-scale path testable in simulation: the pad's
    true size and the true altitude are both known, so the altitude the
    autopilot infers from ``ref_size_px`` can be scored against ground truth.

    The pad is flat on the floor, so it also clears the relief mask underneath
    itself -- a printed sheet does not sit on top of the furniture.
    """
    path = Path(image_path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"pad image not readable: {path}")

    mpp = float(texture.m_per_px)
    pad_w_px = max(1, int(round(float(size_m[0]) / mpp)))
    pad_h_px = max(1, int(round(float(size_m[1]) / mpp)))
    pad = cv2.resize(img, (pad_w_px, pad_h_px), interpolation=cv2.INTER_AREA)

    gray = texture.gray.copy()
    mask = None if texture.relief_mask is None else texture.relief_mask.copy()
    h, w = gray.shape[:2]
    cu, cv_ = texture.world_to_texture_px(float(center_m[0]), float(center_m[1]))
    u0 = int(round(cu - pad_w_px * 0.5))
    v0 = int(round(cv_ - pad_h_px * 0.5))

    su0, sv0 = max(0, -u0), max(0, -v0)
    du0, dv0 = max(0, u0), max(0, v0)
    du1, dv1 = min(w, u0 + pad_w_px), min(h, v0 + pad_h_px)
    if du1 <= du0 or dv1 <= dv0:
        raise ValueError("pad lies entirely outside the ground texture")

    gray[dv0:dv1, du0:du1] = pad[sv0 : sv0 + (dv1 - dv0), su0 : su0 + (du1 - du0)]
    if mask is not None:
        mask[dv0:dv1, du0:du1] = 0.0

    return GroundTexture(
        gray=gray,
        m_per_px=mpp,
        relief_mask=mask,
        relief_height_m=float(texture.relief_height_m),
    )
