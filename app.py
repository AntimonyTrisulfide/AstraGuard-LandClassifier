"""AstraGuard land-cover segmentation demo. Run with: streamlit run app.py"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st

from astraguard_landcover.classes import CLASS_COLORS, CLASS_NAMES, IGNORE_INDEX
from astraguard_landcover.predict import predict_raster

st.set_page_config(page_title="AstraGuard Land-Cover Intelligence", page_icon="AG", layout="wide")
COLOR_MAP = np.asarray([CLASS_COLORS[i][:3] for i in range(len(CLASS_NAMES))], dtype=np.uint8)


def rgb_preview(data: np.ndarray) -> np.ndarray:
    """Create an RGB display from B04/B03/B02 with robust percentile stretching."""
    rgb = np.nan_to_num(data[[2, 1, 0]].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low = np.percentile(rgb, 2, axis=(1, 2), keepdims=True)
    high = np.percentile(rgb, 98, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1)
    return np.moveaxis(rgb, 0, -1)


def render_mask(mask: np.ndarray) -> np.ndarray:
    rendered = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != IGNORE_INDEX
    rendered[valid] = COLOR_MAP[mask[valid]]
    return rendered


def run_inference(checkpoint: Path, uploaded, server_input: Path | None, tile_size: int, overlap: int, batch_size: int):
    with tempfile.TemporaryDirectory(prefix="astraguard_demo_") as tmp:
        tmp_path = Path(tmp)
        input_path = server_input
        if uploaded is not None:
            input_path = tmp_path / uploaded.name
            input_path.write_bytes(uploaded.getvalue())
        output_path = tmp_path / "prediction.tif"
        if input_path is None or not input_path.is_file():
            raise FileNotFoundError("Choose an uploaded GeoTIFF or a valid server-side GeoTIFF path.")
        result = predict_raster(
            checkpoint_path=checkpoint, input_path=input_path, output_path=output_path,
            tile_size=tile_size, overlap=overlap, batch_size=batch_size, max_pixels=50_000_000,
        )
        with rasterio.open(input_path) as source:
            source_data = source.read()
        with rasterio.open(output_path) as prediction:
            mask = prediction.read(1)
    return source_data, mask, result


st.title("AstraGuard Land-Cover Intelligence")
st.caption("Sentinel-2 six-band segmentation | DeepLabV3+ ResNet-34 | Madhya Pradesh")

with st.sidebar:
    st.header("Inference settings")
    checkpoint_text = st.text_input(
        "best.pt path", value=os.environ.get("ASTRAGUARD_CHECKPOINT", ""),
        help="Path visible from the machine running Streamlit.",
    )
    tile_size = st.selectbox("Tile size", [256], index=0)
    overlap = st.slider("Window overlap", 0, 128, 64, 16)
    batch_size = st.slider("Inference batch size", 1, 16, 4)
    st.divider()
    st.markdown("**Required bands**")
    st.code("B02 B03 B04 B08 B11 B12")

uploaded = st.file_uploader("Upload a six-band Sentinel-2 GeoTIFF", type=["tif", "tiff"])
server_input_text = st.text_input(
    "Or use a server-side GeoTIFF path",
    value=os.environ.get("ASTRAGUARD_DEMO_INPUT", ""),
    help="Useful on HPC when the raw AOI already exists on the shared filesystem.",
)
server_input = Path(server_input_text).expanduser() if server_input_text else None
if not uploaded and server_input is None:
    st.info("Upload a small GeoTIFF tile or enter a server-side GeoTIFF path.")
    st.stop()

checkpoint = Path(checkpoint_text).expanduser()
if not checkpoint.is_file():
    st.error("Checkpoint not found. Set the path to best.pt in the sidebar or ASTRAGUARD_CHECKPOINT.")
    st.stop()

with st.spinner("Running sliding-window segmentation..."):
    try:
        source_data, mask, result = run_inference(checkpoint, uploaded, server_input, tile_size, overlap, batch_size)
    except Exception as exc:
        st.exception(exc)
        st.stop()

st.success("Prediction complete")
left, right = st.columns(2)
with left:
    st.subheader("Sentinel-2 preview")
    st.image(rgb_preview(source_data), use_container_width=True, caption="False-colour display: B04 / B03 / B02")
with right:
    st.subheader("Predicted land cover")
    st.image(render_mask(mask), use_container_width=True, caption="Model output; colours follow the shared class contract")

st.subheader("Area summary")
cols = st.columns(len(CLASS_NAMES))
for column, class_name in zip(cols, CLASS_NAMES):
    details = result["classes"][class_name]
    with column:
        st.metric(class_name.replace("_", " ").title(), f"{details['pixels']:,} px")
        if details["area_km2"] is not None:
            st.caption(f"{details['area_km2']:.2f} km²")

valid_pixels = int(np.sum(mask != IGNORE_INDEX))
input_label = uploaded.name if uploaded is not None else str(server_input)
st.caption(f"Input: {input_label} | Valid pixels: {valid_pixels:,} | CRS: {result.get('crs') or 'not available'}")
fig, ax = plt.subplots(figsize=(9, 1.2))
ax.barh(CLASS_NAMES, [result["classes"][name]["pixels"] for name in CLASS_NAMES], color=COLOR_MAP / 255.0)
ax.set_xlabel("Pixels")
ax.set_title("Predicted class distribution")
fig.tight_layout()
st.pyplot(fig, clear_figure=True)
