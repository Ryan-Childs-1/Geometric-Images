import json
import math
import hashlib
from dataclasses import dataclass
from typing import Dict

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.tri as mtri


# -----------------------------
# Utility helpers
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def to_uint8(img_f: np.ndarray) -> np.ndarray:
    img_f = np.clip(img_f, 0.0, 255.0)
    return img_f.astype(np.uint8)


def img_to_np_rgb(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    return np.array(pil_img)


def np_to_pil(img_np: np.ndarray) -> Image.Image:
    return Image.fromarray(to_uint8(img_np), mode="RGB")


def resize_keep_aspect_pil(pil_img: Image.Image, max_side: int) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    if h >= w:
        new_h = max_side
        new_w = int(round(w * (max_side / h)))
    else:
        new_w = max_side
        new_h = int(round(h * (max_side / w)))
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def stable_hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def pil_image_bytes(pil_img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# -----------------------------
# Geometry model
# -----------------------------
@dataclass
class ShapeEncoding:
    width: int
    height: int
    points_xy: np.ndarray  # (N,2) float in pixel coords
    triangles: np.ndarray  # (M,3) int indices
    tri_colors: np.ndarray # (M,3) uint8 RGB

    def to_json_dict(self) -> Dict:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "points_xy": self.points_xy.tolist(),
            "triangles": self.triangles.tolist(),
            "tri_colors": self.tri_colors.tolist(),
        }

    @staticmethod
    def from_json_dict(d: Dict) -> "ShapeEncoding":
        return ShapeEncoding(
            width=int(d["width"]),
            height=int(d["height"]),
            points_xy=np.array(d["points_xy"], dtype=np.float32),
            triangles=np.array(d["triangles"], dtype=np.int32),
            tri_colors=np.array(d["tri_colors"], dtype=np.uint8),
        )


# -----------------------------
# Simple edge detection (Sobel) in pure numpy
# -----------------------------
def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    # ITU-R BT.601
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    return (0.299 * r + 0.587 * g + 0.114 * b)


def conv3(gray: np.ndarray, k: np.ndarray) -> np.ndarray:
    # 3x3 convolution via slicing (fast enough for moderate sizes)
    p = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    out = (
        k[0, 0] * p[:-2, :-2] + k[0, 1] * p[:-2, 1:-1] + k[0, 2] * p[:-2, 2:] +
        k[1, 0] * p[1:-1, :-2] + k[1, 1] * p[1:-1, 1:-1] + k[1, 2] * p[1:-1, 2:] +
        k[2, 0] * p[2:, :-2] + k[2, 1] * p[2:, 1:-1] + k[2, 2] * p[2:, 2:]
    )
    return out


def sobel_edges(gray: np.ndarray, threshold: float) -> np.ndarray:
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    gx = conv3(gray, kx)
    gy = conv3(gray, ky)
    mag = np.sqrt(gx * gx + gy * gy)
    # normalize to 0..255-ish scale
    mag = mag / (mag.max() + 1e-6) * 255.0
    edges = (mag >= threshold).astype(np.uint8)
    return edges


# -----------------------------
# Point sampling (canonical geometry)
# -----------------------------
def sample_points_canonical(
    img_rgb: np.ndarray,
    n_points: int,
    edge_weight: float,
    edge_thresh: float,
    border_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img_rgb.shape[:2]

    gray = rgb_to_gray(img_rgb)
    edges = sobel_edges(gray, threshold=edge_thresh)
    ys, xs = np.where(edges > 0)
    edge_coords = np.stack([xs, ys], axis=1) if len(xs) else np.zeros((0, 2), dtype=np.int32)

    n_edge = int(round(n_points * clamp(edge_weight, 0.0, 1.0)))
    n_rand = max(0, n_points - n_edge)

    pts = []

    if edge_coords.shape[0] > 0 and n_edge > 0:
        take = min(n_edge, edge_coords.shape[0])
        idx = rng.choice(edge_coords.shape[0], size=take, replace=False)
        pts.append(edge_coords[idx].astype(np.float32))
        if take < n_edge:
            pad = n_edge - take
            xr = rng.uniform(0, w - 1, size=(pad, 1))
            yr = rng.uniform(0, h - 1, size=(pad, 1))
            pts.append(np.concatenate([xr, yr], axis=1).astype(np.float32))
    else:
        if n_edge > 0:
            xr = rng.uniform(0, w - 1, size=(n_edge, 1))
            yr = rng.uniform(0, h - 1, size=(n_edge, 1))
            pts.append(np.concatenate([xr, yr], axis=1).astype(np.float32))

    if n_rand > 0:
        xr = rng.uniform(0, w - 1, size=(n_rand, 1))
        yr = rng.uniform(0, h - 1, size=(n_rand, 1))
        pts.append(np.concatenate([xr, yr], axis=1).astype(np.float32))

    pts = np.concatenate(pts, axis=0) if len(pts) else np.zeros((0, 2), dtype=np.float32)

    # Border points for stability
    border_points = max(4, int(border_points))
    top = np.linspace(0, w - 1, border_points, dtype=np.float32)
    bot = np.linspace(0, w - 1, border_points, dtype=np.float32)
    lef = np.linspace(0, h - 1, border_points, dtype=np.float32)
    rig = np.linspace(0, h - 1, border_points, dtype=np.float32)

    border = np.concatenate([
        np.stack([top, np.zeros_like(top)], axis=1),
        np.stack([bot, np.full_like(bot, h - 1)], axis=1),
        np.stack([np.zeros_like(lef), lef], axis=1),
        np.stack([np.full_like(rig, w - 1), rig], axis=1),
    ], axis=0)

    pts = np.concatenate([pts, border], axis=0)

    # Deduplicate on integer grid
    q = np.round(pts).astype(np.int32)
    _, uniq_idx = np.unique(q, axis=0, return_index=True)
    pts = pts[uniq_idx].astype(np.float32)

    # Ensure enough
    target = n_points + border.shape[0]
    if pts.shape[0] < target:
        need = target - pts.shape[0]
        xr = rng.uniform(0, w - 1, size=(need, 1))
        yr = rng.uniform(0, h - 1, size=(need, 1))
        pts = np.concatenate([pts, np.concatenate([xr, yr], axis=1).astype(np.float32)], axis=0)

    return pts


# -----------------------------
# Triangulation + rendering
# -----------------------------
def build_triangulation(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] < 3:
        return np.zeros((0, 3), dtype=np.int32)
    x = points_xy[:, 0].astype(np.float64)
    y = points_xy[:, 1].astype(np.float64)
    tri = mtri.Triangulation(x, y)
    return tri.triangles.astype(np.int32)


def triangle_centroids(points_xy: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    pts = points_xy[triangles]
    return pts.mean(axis=1)


def sample_colors_at_centroids(img_rgb: np.ndarray, centroids_xy: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    xs = np.clip(np.round(centroids_xy[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(centroids_xy[:, 1]).astype(np.int32), 0, h - 1)
    return img_rgb[ys, xs].astype(np.uint8)


def tri_colors_for_image(img_rgb: np.ndarray, points_xy: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    if triangles.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    centroids = triangle_centroids(points_xy, triangles)
    return sample_colors_at_centroids(img_rgb, centroids)


def blend_colors(c1: np.ndarray, c2: np.ndarray, alpha: float) -> np.ndarray:
    a = float(alpha)
    out = (1.0 - a) * c1.astype(np.float32) + a * c2.astype(np.float32)
    return to_uint8(out)


def render_triangles_pil(
    width: int,
    height: int,
    points_xy: np.ndarray,
    triangles: np.ndarray,
    tri_colors: np.ndarray,
    draw_wire: bool,
    wire_thickness: int,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas, "RGB")

    for t_idx, tri in enumerate(triangles):
        p = points_xy[tri]
        poly = [(float(p[0, 0]), float(p[0, 1])),
                (float(p[1, 0]), float(p[1, 1])),
                (float(p[2, 0]), float(p[2, 1]))]
        c = tuple(int(x) for x in tri_colors[t_idx])
        draw.polygon(poly, fill=c)

    if draw_wire and triangles.shape[0] > 0:
        wt = max(1, int(wire_thickness))
        for tri in triangles:
            p = points_xy[tri]
            a = (float(p[0, 0]), float(p[0, 1]))
            b = (float(p[1, 0]), float(p[1, 1]))
            c = (float(p[2, 0]), float(p[2, 1]))
            draw.line([a, b], fill=(255, 255, 255), width=wt)
            draw.line([b, c], fill=(255, 255, 255), width=wt)
            draw.line([c, a], fill=(255, 255, 255), width=wt)

    return canvas


# -----------------------------
# Geometry editing operations
# -----------------------------
def clip_points_to_image(points_xy: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = points_xy.copy().astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, w - 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, h - 1.0)
    return pts


def apply_affine(points_xy: np.ndarray, cx: float, cy: float, scale: float, rot_deg: float, tx: float, ty: float) -> np.ndarray:
    pts = points_xy.copy().astype(np.float32)
    pts[:, 0] -= cx
    pts[:, 1] -= cy
    theta = math.radians(rot_deg)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]], dtype=np.float32)
    pts = (pts @ R.T) * float(scale)
    pts[:, 0] += cx + float(tx)
    pts[:, 1] += cy + float(ty)
    return pts


def jitter_points(points_xy: np.ndarray, amount: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = points_xy.copy().astype(np.float32)
    noise = rng.normal(0.0, float(amount), size=pts.shape).astype(np.float32)
    return pts + noise


def neighbor_relax(points_xy: np.ndarray, w: int, h: int, iters: int = 1) -> np.ndarray:
    """
    Simple relaxation using Delaunay adjacency (no Voronoi, no scipy).
    """
    pts = points_xy.copy().astype(np.float32)
    if pts.shape[0] < 3:
        return pts

    for _ in range(max(0, int(iters))):
        tris = build_triangulation(pts)
        n = pts.shape[0]
        neigh = [[] for _ in range(n)]
        for a, b, c in tris:
            neigh[a].extend([b, c])
            neigh[b].extend([a, c])
            neigh[c].extend([a, b])

        new_pts = pts.copy()
        for i in range(n):
            if not neigh[i]:
                continue
            nb = np.array(neigh[i], dtype=np.int32)
            m = pts[nb].mean(axis=0)
            new_pts[i] = 0.75 * pts[i] + 0.25 * m

        pts = clip_points_to_image(new_pts, w, h)

    return pts


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Image → Geometric Shape Encoder", layout="wide")
st.title("Image → Geometric Shape Encoder (No OpenCV, Editable, Shared Geometry)")

with st.expander("What this app does", expanded=True):
    st.markdown(
        """
- Converts images into a **shared geometric representation** using a canonical point set + **Delaunay triangulation**.
- All images are “related” because the **triangles are the same**; only **triangle colors** change per image.
- You can **edit the shape** (move/add/delete/relax/jitter/affine) and see the render update immediately.
- Optional **blend** between two images in the same geometry.
- Export/import as JSON + download PNG.
        """
    )

# Session state init
if "images" not in st.session_state:
    st.session_state.images = {}  # key -> dict(name,img,w,h)
if "active_key" not in st.session_state:
    st.session_state.active_key = None
if "canonical_points" not in st.session_state:
    st.session_state.canonical_points = None
if "edit_points" not in st.session_state:
    st.session_state.edit_points = None
if "triangles" not in st.session_state:
    st.session_state.triangles = None

# Sidebar upload
st.sidebar.header("1) Upload Images")
uploads = st.sidebar.file_uploader(
    "Upload one or more images",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

max_side = st.sidebar.slider("Max display size (px)", 256, 1600, 900, 50)

if uploads:
    for f in uploads:
        raw = f.read()
        pil = Image.open(f).convert("RGB")
        pil = resize_keep_aspect_pil(pil, max_side=max_side)
        np_img = img_to_np_rgb(pil)
        key = stable_hash_bytes(raw)
        st.session_state.images[key] = {
            "name": f.name,
            "img": np_img,
            "w": np_img.shape[1],
            "h": np_img.shape[0],
        }
        if st.session_state.active_key is None:
            st.session_state.active_key = key

keys = list(st.session_state.images.keys())
if not keys:
    st.info("Upload at least one image to begin.")
    st.stop()

name_map = {k: st.session_state.images[k]["name"] for k in keys}
active = st.sidebar.selectbox("Active image", options=keys, format_func=lambda k: name_map.get(k, k))
st.session_state.active_key = active

img = st.session_state.images[active]["img"]
h, w = img.shape[:2]

# Geometry controls
st.sidebar.divider()
st.sidebar.header("2) Canonical Geometry (Shared Across Images)")
n_points = st.sidebar.slider("Base points (not counting border)", 50, 2500, 500, 25)
edge_weight = st.sidebar.slider("Edge emphasis", 0.0, 1.0, 0.65, 0.05)
edge_thresh = st.sidebar.slider("Edge threshold (Sobel)", 1.0, 255.0, 70.0, 1.0)
border_pts = st.sidebar.slider("Border points per edge", 4, 200, 35, 1)
seed = st.sidebar.number_input("Seed", value=7, step=1)
build_btn = st.sidebar.button("Build / Rebuild Canonical Geometry", use_container_width=True)

if build_btn or (st.session_state.canonical_points is None) or (st.session_state.edit_points is None):
    pts = sample_points_canonical(
        img_rgb=img,
        n_points=int(n_points),
        edge_weight=float(edge_weight),
        edge_thresh=float(edge_thresh),
        border_points=int(border_pts),
        seed=int(seed),
    )
    pts = clip_points_to_image(pts, w, h)
    st.session_state.canonical_points = pts
    st.session_state.edit_points = pts.copy()

# Edit controls
st.sidebar.divider()
st.sidebar.header("3) Edit Geometry")
pts = st.session_state.edit_points
if pts is None or pts.shape[0] < 3:
    st.warning("Not enough points to triangulate.")
    st.stop()

st.sidebar.subheader("Point edit (select + move)")
point_idx = st.sidebar.number_input("Point index", min_value=0, max_value=max(0, pts.shape[0]-1), value=0, step=1)
curx, cury = float(pts[int(point_idx), 0]), float(pts[int(point_idx), 1])
newx = st.sidebar.slider("X", 0.0, float(w-1), curx, 1.0)
newy = st.sidebar.slider("Y", 0.0, float(h-1), cury, 1.0)
if (newx != curx) or (newy != cury):
    pts2 = pts.copy()
    pts2[int(point_idx)] = [float(newx), float(newy)]
    st.session_state.edit_points = pts2
    pts = pts2

st.sidebar.subheader("Add / delete point")
add_x = st.sidebar.slider("Add point X", 0.0, float(w-1), float(w/2), 1.0)
add_y = st.sidebar.slider("Add point Y", 0.0, float(h-1), float(h/2), 1.0)
c1, c2 = st.sidebar.columns(2)
if c1.button("Add point", use_container_width=True):
    pts2 = np.vstack([pts, np.array([[add_x, add_y]], dtype=np.float32)])
    st.session_state.edit_points = clip_points_to_image(pts2, w, h)
    pts = st.session_state.edit_points
if c2.button("Delete selected", use_container_width=True):
    if pts.shape[0] > 3:
        mask = np.ones((pts.shape[0],), dtype=bool)
        mask[int(point_idx)] = False
        st.session_state.edit_points = pts[mask]
        pts = st.session_state.edit_points

st.sidebar.subheader("Relax / jitter / transform")
relax_iters = st.sidebar.slider("Relax iterations", 0, 25, 3, 1)
jitter_amt = st.sidebar.slider("Jitter amount (px)", 0.0, 50.0, 0.0, 0.5)
r1, r2 = st.sidebar.columns(2)
if r1.button("Relax", use_container_width=True):
    st.session_state.edit_points = neighbor_relax(st.session_state.edit_points, w, h, iters=int(relax_iters))
    pts = st.session_state.edit_points
if r2.button("Jitter", use_container_width=True):
    st.session_state.edit_points = clip_points_to_image(
        jitter_points(st.session_state.edit_points, amount=float(jitter_amt), seed=int(seed)+123),
        w, h
    )
    pts = st.session_state.edit_points

st.sidebar.subheader("Global affine")
scale = st.sidebar.slider("Scale", 0.25, 2.5, 1.0, 0.01)
rot = st.sidebar.slider("Rotate (deg)", -180.0, 180.0, 0.0, 1.0)
tx = st.sidebar.slider("Translate X", -float(w), float(w), 0.0, 1.0)
ty = st.sidebar.slider("Translate Y", -float(h), float(h), 0.0, 1.0)
if st.sidebar.button("Apply affine to all points", use_container_width=True):
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    pts2 = apply_affine(pts, cx, cy, float(scale), float(rot), float(tx), float(ty))
    st.session_state.edit_points = clip_points_to_image(pts2, w, h)
    pts = st.session_state.edit_points

if st.sidebar.button("Reset geometry to canonical", use_container_width=True):
    st.session_state.edit_points = st.session_state.canonical_points.copy()
    pts = st.session_state.edit_points

# Build triangles every run (point edits change connectivity)
tris = build_triangulation(pts)
st.session_state.triangles = tris

# Colors for active image
active_colors = tri_colors_for_image(img, pts, tris)

# Rendering options
st.sidebar.divider()
st.sidebar.header("4) Render / Blend / Export")
draw_wire = st.sidebar.checkbox("Draw wireframe", value=False)
wire_thickness = st.sidebar.slider("Wire thickness", 1, 6, 1, 1)
show_points = st.sidebar.checkbox("Show points overlay", value=True)
point_radius = st.sidebar.slider("Point radius", 1, 8, 2, 1)

# Blend option
blend_on = False
blend_key = None
alpha = 0.0
if len(keys) >= 2:
    st.sidebar.subheader("Blend between two images")
    blend_on = st.sidebar.checkbox("Enable blending", value=False)
    if blend_on:
        blend_key = st.sidebar.selectbox(
            "Blend with",
            options=[k for k in keys if k != active],
            format_func=lambda k: st.session_state.images[k]["name"]
        )
        alpha = st.sidebar.slider("Blend alpha", 0.0, 1.0, 0.5, 0.01)

if blend_on and blend_key is not None:
    img2 = st.session_state.images[blend_key]["img"]
    # resize img2 to match active image size using PIL
    if img2.shape[:2] != img.shape[:2]:
        pil2 = np_to_pil(img2).resize((w, h), Image.Resampling.LANCZOS)
        img2r = img_to_np_rgb(pil2)
    else:
        img2r = img2
    colors2 = tri_colors_for_image(img2r, pts, tris)
    colors = blend_colors(active_colors, colors2, alpha=float(alpha))
else:
    colors = active_colors

enc = ShapeEncoding(width=w, height=h, points_xy=pts.astype(np.float32), triangles=tris.astype(np.int32), tri_colors=colors.astype(np.uint8))
geom_pil = render_triangles_pil(w, h, enc.points_xy, enc.triangles, enc.tri_colors, draw_wire=draw_wire, wire_thickness=wire_thickness)

# Points overlay
if show_points:
    overlay = geom_pil.copy()
    d = ImageDraw.Draw(overlay, "RGB")
    r = int(point_radius)
    for i, (x, y) in enumerate(enc.points_xy):
        fill = (255, 0, 0) if i == int(point_idx) else (0, 255, 0)
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    geom_pil = overlay

# Layout
left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("Original")
    st.image(img, use_container_width=True)
    st.caption(f"{st.session_state.images[active]['name']} — {w}×{h}")

with right:
    st.subheader("Geometric Encoding (Editable)")
    st.image(geom_pil, use_container_width=True)
    st.caption(f"Points: {pts.shape[0]} | Triangles: {tris.shape[0]}")

# Export/Import
st.divider()
cA, cB, cC = st.columns([1, 1, 1], gap="large")

with cA:
    st.subheader("Export shape encoding (JSON)")
    enc_json = json.dumps(enc.to_json_dict(), indent=2)
    st.download_button(
        "Download JSON",
        data=enc_json.encode("utf-8"),
        file_name="shape_encoding.json",
        mime="application/json",
        use_container_width=True,
    )

with cB:
    st.subheader("Download rendered PNG")
    png_bytes = pil_image_bytes(geom_pil)
    st.download_button(
        "Download PNG",
        data=png_bytes,
        file_name="geometric_render.png",
        mime="image/png",
        use_container_width=True,
    )

with cC:
    st.subheader("Import encoding (JSON)")
    up_json = st.file_uploader("Upload shape_encoding.json", type=["json"], accept_multiple_files=False)
    if up_json is not None:
        try:
            d = json.loads(up_json.read().decode("utf-8"))
            imported = ShapeEncoding.from_json_dict(d)
            st.session_state.edit_points = clip_points_to_image(imported.points_xy, w, h)
            st.success("Imported geometry loaded into editor (points updated).")
        except Exception as e:
            st.error(f"Failed to import JSON: {e}")
