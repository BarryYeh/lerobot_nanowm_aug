#!/usr/bin/env python
"""Video to 3D Point Cloud Pipeline using Depth Anything 3.

Extracts frames from an MP4 video, estimates depth and camera poses
using DA3 multi-view inference, and generates a colored 3D point cloud
saved as PLY. Optionally launches a viser interactive 3D viewer.

Usage:
    python scripts/video_to_pointcloud.py \\
        --video sample_videos/train_sample_0.mp4 \\
        --output output/scene.ply \\
        --max_frames 10 \\
        --visualize
"""

import argparse
import struct
import time
from pathlib import Path

import imageio
import numpy as np
from PIL import Image


def extract_frames(video_path: str, max_frames: int, frame_step: int) -> list:
    """Extract frames from a video file.

    Args:
        video_path: Path to the input MP4 video.
        max_frames: Maximum number of frames to extract.
        frame_step: Extract every Nth frame.

    Returns:
        List of [H, W, 3] uint8 numpy arrays.
    """
    reader = imageio.get_reader(video_path)
    frames = []
    for i, frame in enumerate(reader):
        if i % frame_step != 0:
            continue
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    reader.close()

    if len(frames) < 2:
        raise ValueError(
            f"Need at least 2 frames for multi-view depth estimation, "
            f"got {len(frames)}. Try reducing --frame_step or using a longer video."
        )

    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames


def restore_native_resolution(frames: list, native_h: int, native_w: int) -> list:
    """Restore frames to their native aspect ratio before DA3 inference.

    Rollout videos are typically generated at a square resolution (e.g.,
    256x256) by stretching the original frames. This distorts the aspect
    ratio and breaks DA3's depth estimation. This function resizes frames
    back to the native resolution to restore correct geometry.

    Follows the same pattern as inference_engine.py:resize_for_display().

    Args:
        frames: List of [H, W, 3] uint8 numpy arrays (stretched).
        native_h: Native height in pixels.
        native_w: Native width in pixels.

    Returns:
        List of resized [native_h, native_w, 3] uint8 numpy arrays.
    """
    H, W = frames[0].shape[:2]
    if H == native_h and W == native_w:
        print(f"Frames already at native resolution {W}x{H}, skipping resize")
        return frames

    resized = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = img.resize((native_w, native_h), Image.BILINEAR)
        resized.append(np.array(img))

    print(f"Restored native resolution: {W}x{H} -> {native_w}x{native_h}")
    return resized


def load_da3_model(model_id: str, device: str):
    """Load Depth Anything 3 model.

    Args:
        model_id: HuggingFace model ID (e.g., 'depth-anything/DA3-LARGE-1.1').
        device: Inference device ('cuda' or 'cpu').

    Returns:
        Loaded DA3 model on the specified device.
    """
    from depth_anything_3.api import DepthAnything3

    print(f"Loading DA3 model: {model_id}")
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)
    print(f"Model loaded on {device}")
    return model


def run_inference(model, frames, export_dir, process_res, conf_threshold, max_points):
    """Run DA3 multi-view inference on extracted frames.

    Args:
        model: Loaded DA3 model.
        frames: List of [H, W, 3] uint8 numpy arrays.
        export_dir: Directory for DA3 native exports.
        process_res: DA3 internal processing resolution.
        conf_threshold: Confidence percentile for point filtering.
        max_points: Maximum number of points in output.

    Returns:
        DA3 prediction object with depth, conf, intrinsics, extrinsics.
    """
    print(f"Running DA3 inference on {len(frames)} frames...")

    Path(export_dir).mkdir(parents=True, exist_ok=True)

    prediction = model.inference(
        image=frames,
        process_res=process_res,
        conf_thresh_percentile=conf_threshold,
        num_max_points=max_points,
        export_dir=export_dir,
        export_format="mini_npz",
    )

    print("Inference complete")
    return prediction


def extrinsics_to_4x4(ext: np.ndarray) -> np.ndarray:
    """Convert a 3x4 extrinsic matrix to 4x4 by appending [0, 0, 0, 1].

    DA3 returns extrinsics as (N, 3, 4) w2c matrices. Many operations
    (e.g., np.linalg.inv) require a full 4x4 matrix.

    Args:
        ext: [3, 4] or [4, 4] extrinsic matrix.

    Returns:
        [4, 4] extrinsic matrix.
    """
    if ext.shape == (4, 4):
        return ext
    mat = np.eye(4, dtype=ext.dtype)
    mat[:3, :] = ext
    return mat


def unproject_frame(
    depth: np.ndarray,
    conf: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image: np.ndarray,
    conf_percentile: float,
):
    """Unproject a single frame's depth map to 3D world coordinates.

    Args:
        depth: [H, W] float32 depth map.
        conf: [H, W] float32 confidence map.
        intrinsics: [3, 3] camera intrinsic matrix.
        extrinsics: [3, 4] or [4, 4] world-to-camera transformation matrix.
        image: [H, W, 3] uint8 RGB image.
        conf_percentile: Confidence percentile threshold for filtering.

    Returns:
        Tuple of (points [N, 3] float32, colors [N, 3] uint8).
    """
    H, W = depth.shape

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Extract intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Unproject to camera coordinates
    X_cam = (u - cx) * depth / fx
    Y_cam = (v - cy) * depth / fy
    Z_cam = depth

    # Stack into (N, 3) array
    pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1).reshape(-1, 3)

    # Transform to world coordinates: c2w = inv(w2c)
    c2w = np.linalg.inv(extrinsics_to_4x4(extrinsics))
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = (R @ pts_cam.T).T + t

    # Get colors from image
    colors = image.reshape(-1, 3)

    # Filter by confidence threshold (percentile-based)
    conf_flat = conf.reshape(-1)
    threshold = np.percentile(conf_flat, conf_percentile)
    mask = conf_flat >= threshold

    # Also filter out zero/negative depth
    depth_flat = depth.reshape(-1)
    mask = mask & (depth_flat > 0)

    return pts_world[mask].astype(np.float32), colors[mask]


def save_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """Save colored point cloud as binary PLY file.

    Args:
        path: Output PLY file path.
        points: [N, 3] float32 array of XYZ coordinates.
        colors: [N, 3] uint8 array of RGB colors.
    """
    N = len(points)
    if N == 0:
        raise ValueError("No points to save. Try lowering --conf_threshold.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    # Use numpy structured array for efficient binary writing
    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices = np.empty(N, dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)

    print(f"Saved {N:,} points to {path}")


def load_ply(path: str):
    """Load a binary PLY point cloud file.

    Args:
        path: Path to PLY file.

    Returns:
        Tuple of (points [N, 3] float32, colors [N, 3] uint8).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    with open(path, "rb") as f:
        # Parse header
        num_vertices = None
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            if line == "end_header":
                break

        if num_vertices is None:
            raise ValueError(f"Could not parse vertex count from PLY header: {path}")

        # Read binary data
        vertex_dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        vertices = np.frombuffer(f.read(num_vertices * vertex_dtype.itemsize), dtype=vertex_dtype)

    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1)
    colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=-1)
    print(f"Loaded {len(points):,} points from {path}")
    return points, colors


def launch_viewer_ply(points, colors):
    """Launch viser viewer for a standalone PLY point cloud (no camera info).

    Args:
        points: [N, 3] point coordinates.
        colors: [N, 3] point colors (uint8).
    """
    import viser

    server = viser.ViserServer()

    server.scene.add_point_cloud(
        "/points",
        points=points,
        colors=colors,
        point_size=0.005,
        point_shape="circle",
    )

    with server.gui.add_folder("Controls"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.05, step=0.001, initial_value=0.005
        )

    @gui_point_size.on_update
    def _(_) -> None:
        server.scene.add_point_cloud(
            "/points",
            points=points,
            colors=colors,
            point_size=gui_point_size.value,
            point_shape="circle",
        )

    url = f"http://localhost:{server.get_port()}"
    print(f"\nViewer running at: {url}")
    print("Press Ctrl+C to exit.")

    while True:
        time.sleep(1.0)


def save_viewer_data(path: str, prediction, conf_threshold: float):
    """Save all viewer data to a single .npz file for offline viewing.

    Stores depth maps, confidence, camera parameters, and processed images
    so the rich viser viewer can be launched later without re-running inference.

    Args:
        path: Output .npz file path.
        prediction: DA3 prediction object.
        conf_threshold: Confidence percentile used for filtering.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        depth=prediction.depth,
        conf=prediction.conf,
        intrinsics=prediction.intrinsics,
        extrinsics=prediction.extrinsics,
        processed_images=prediction.processed_images,
        conf_threshold=np.array(conf_threshold),
    )
    print(f"Saved viewer data to {path}")


def load_viewer_data(path: str) -> dict:
    """Load viewer data from a .npz file.

    Args:
        path: Path to .npz file saved by save_viewer_data().

    Returns:
        Dict with keys: depth, conf, intrinsics, extrinsics,
        processed_images, conf_threshold.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Viewer data not found: {path}")

    data = np.load(path)
    result = {
        "depth": data["depth"],
        "conf": data["conf"],
        "intrinsics": data["intrinsics"],
        "extrinsics": data["extrinsics"],
        "processed_images": data["processed_images"],
        "conf_threshold": float(data["conf_threshold"]),
    }
    print(f"Loaded viewer data from {path} ({len(result['depth'])} frames)")
    return result


def save_depth_visualizations(prediction, frames, output_dir: str):
    """Save side-by-side depth and confidence visualizations.

    For each frame, creates a figure with three panels:
    original image | depth map (turbo colormap) | confidence map.

    Args:
        prediction: DA3 prediction object.
        frames: List of [H, W, 3] uint8 numpy arrays (original frames).
        output_dir: Directory to save visualization PNGs.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_frames = len(frames)
    for i in range(num_frames):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original frame (use processed image to match depth map dimensions)
        axes[0].imshow(prediction.processed_images[i])
        axes[0].set_title(f"Frame {i}")
        axes[0].axis("off")

        # Depth map
        im_depth = axes[1].imshow(prediction.depth[i], cmap="turbo")
        axes[1].set_title("Depth")
        axes[1].axis("off")
        plt.colorbar(im_depth, ax=axes[1], fraction=0.046, pad=0.04)

        # Confidence map
        im_conf = axes[2].imshow(prediction.conf[i], cmap="viridis")
        axes[2].set_title("Confidence")
        axes[2].axis("off")
        plt.colorbar(im_conf, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = output_dir / f"frame_{i:04d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {num_frames} depth visualizations to {output_dir}")


def rotation_matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion in (w, x, y, z) format.

    Uses Shepperd's method for numerical stability.

    Args:
        R: [3, 3] rotation matrix.

    Returns:
        [4] quaternion array in (w, x, y, z) order.
    """
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z], dtype=np.float64)


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """Render depth map as a turbo-colormap uint8 image.

    Args:
        depth: [H, W] float32 depth map.

    Returns:
        [H, W, 3] uint8 RGB image.
    """
    import matplotlib.cm as cm

    d = depth.copy()
    valid = d > 0
    if valid.any():
        d_min, d_max = d[valid].min(), d[valid].max()
        if d_max > d_min:
            d = (d - d_min) / (d_max - d_min)
        else:
            d[:] = 0.5
    d[~valid] = 0.0
    colored = (cm.turbo(d)[:, :, :3] * 255).astype(np.uint8)
    return colored


def launch_viewer(per_frame_points, per_frame_colors, viewer_data):
    """Launch viser interactive 3D viewer with timeline playback.

    Features:
    - Per-frame point clouds with accumulation mode
    - Camera frustums: current frame highlighted, others dimmed
    - Timeline slider with auto-play
    - RGB and depth map panels in GUI sidebar
    - World-origin axes and ground grid for reference
    - Auto-computed initial camera angle based on scene extent

    Args:
        per_frame_points: List of [N_i, 3] float32 arrays per frame.
        per_frame_colors: List of [N_i, 3] uint8 arrays per frame.
        viewer_data: Dict with keys: depth, conf, intrinsics, extrinsics,
                     processed_images (from prediction or load_viewer_data).
    """
    import viser

    server = viser.ViserServer()
    num_frames = len(per_frame_points)

    depth_maps = viewer_data["depth"]
    intrinsics_arr = viewer_data["intrinsics"]
    extrinsics_arr = viewer_data["extrinsics"]
    processed_images = viewer_data["processed_images"]

    # --- Compute scene geometry for camera init and scale ---
    all_pts = np.concatenate(per_frame_points, axis=0)
    scene_min = all_pts.min(axis=0)
    scene_max = all_pts.max(axis=0)
    centroid = (scene_min + scene_max) / 2.0
    extent = float(np.linalg.norm(scene_max - scene_min))
    frustum_scale = 0.03

    # --- Initial camera position for each connecting client ---
    # DA3 uses OpenCV/COLMAP convention: X-right, Y-down, Z-forward.
    # Real-world "up" is -Y. Place viewer at the first camera's height,
    # slightly behind and above for a near-level perspective.
    first_cam_c2w = np.linalg.inv(extrinsics_to_4x4(extrinsics_arr[0]))
    first_cam_pos = first_cam_c2w[:3, 3]

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (
            float(first_cam_pos[0]),
            float(first_cam_pos[1] - extent * 0.08),  # slightly above (-Y = up)
            float(first_cam_pos[2] - extent * 0.25),   # behind first camera
        )
        client.camera.look_at = tuple(centroid)
        client.camera.up_direction = (0.0, -1.0, 0.0)  # -Y is up in OpenCV

    # Ground grid — warm beige matching Interactive Demo palette (#e5e3dc)
    server.scene.add_grid(
        "/ground",
        plane="xz",
        position=(float(centroid[0]), float(scene_max[1]) + extent * 0.02, float(centroid[2])),
        cell_color=(229, 227, 220),
        section_color=(229, 227, 220),
        cell_size=extent * 0.05,
        section_size=extent * 0.25,
        infinite_grid=True,
    )

    # Pre-render depth colormaps for GUI display
    depth_images = []
    for i in range(num_frames):
        depth_images.append(depth_to_colormap(depth_maps[i]))

    # --- Compute camera poses (c2w) for all frames ---
    cam_positions = []
    cam_wxyz = []
    cam_fov_y = []
    cam_aspect = []
    for i in range(num_frames):
        c2w = np.linalg.inv(extrinsics_to_4x4(extrinsics_arr[i]))
        cam_positions.append(c2w[:3, 3])
        cam_wxyz.append(rotation_matrix_to_wxyz(c2w[:3, :3]))

        K = intrinsics_arr[i]
        H, W = depth_maps[i].shape
        cam_fov_y.append(float(2.0 * np.arctan(H / (2.0 * K[1, 1]))))
        cam_aspect.append(float(W / H))

    # --- Build per-frame scene nodes ---
    frame_nodes = []
    point_nodes = []
    frustum_handles = []

    for i in range(num_frames):
        # Frame node (parent for toggling point cloud visibility)
        frame_node = server.scene.add_frame(
            f"/frames/t{i}", show_axes=False
        )
        frame_nodes.append(frame_node)

        # Point cloud
        point_node = server.scene.add_point_cloud(
            f"/frames/t{i}/points",
            points=per_frame_points[i],
            colors=per_frame_colors[i],
            point_size=0.005,
            point_shape="rounded",
        )
        point_nodes.append(point_node)

    # Camera frustums: separate from frame_nodes so they don't toggle with points
    for i in range(num_frames):
        ds = max(1, max(depth_maps[i].shape) // 200)
        thumbnail = processed_images[i][::ds, ::ds]

        handle = server.scene.add_camera_frustum(
            f"/cameras/frame_{i}",
            fov=cam_fov_y[i],
            aspect=cam_aspect[i],
            scale=frustum_scale,
            image=thumbnail,
            wxyz=cam_wxyz[i],
            position=cam_positions[i],
            color=(60, 60, 60),
        )
        frustum_handles.append(handle)

    # --- GUI controls ---
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
        )
        gui_playing = server.gui.add_checkbox("Playing", initial_value=False)
        gui_fps = server.gui.add_slider(
            "FPS", min=1, max=30, step=1, initial_value=5
        )
        gui_accumulate = server.gui.add_checkbox(
            "Accumulate frames", initial_value=True
        )

    with server.gui.add_folder("Display"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.05, step=0.001, initial_value=0.005
        )
        gui_frustum_scale = server.gui.add_slider(
            "Camera scale", min=0.005, max=0.2,
            step=0.005, initial_value=frustum_scale
        )

    # GUI images: no enclosing folder, with labels, displayed prominently
    server.gui.add_markdown("### Current Frame")
    gui_rgb = server.gui.add_image(processed_images[0], label="RGB")
    gui_depth = server.gui.add_image(depth_images[0], label="Depth")

    # --- Update logic ---
    def update_display():
        current = gui_timestep.value
        accumulate = gui_accumulate.value
        ps = gui_point_size.value

        # Toggle point cloud visibility
        with server.atomic():
            for i in range(num_frames):
                if accumulate:
                    frame_nodes[i].visible = i <= current
                else:
                    frame_nodes[i].visible = i == current

        # Update point sizes
        for i in range(num_frames):
            if frame_nodes[i].visible:
                point_nodes[i].point_size = ps

        # Highlight current frustum, dim others
        fs = gui_frustum_scale.value
        for i in range(num_frames):
            if i == current:
                frustum_handles[i].color = (255, 80, 30)
                frustum_handles[i].line_width = 3.0
            else:
                frustum_handles[i].color = (60, 60, 60)
                frustum_handles[i].line_width = 1.0

        # Update GUI images
        gui_rgb.image = processed_images[current]
        gui_depth.image = depth_images[current]

    # Initial state
    update_display()

    @gui_timestep.on_update
    def _(_) -> None:
        update_display()

    @gui_accumulate.on_update
    def _(_) -> None:
        update_display()

    @gui_point_size.on_update
    def _(_) -> None:
        ps = gui_point_size.value
        for i in range(num_frames):
            if frame_nodes[i].visible:
                point_nodes[i].point_size = ps

    @gui_frustum_scale.on_update
    def _(_) -> None:
        fs = gui_frustum_scale.value
        for i in range(num_frames):
            # Re-create frustums at new scale
            ds = max(1, max(depth_maps[i].shape) // 200)
            thumbnail = processed_images[i][::ds, ::ds]
            is_current = i == gui_timestep.value

            frustum_handles[i] = server.scene.add_camera_frustum(
                f"/cameras/frame_{i}",
                fov=cam_fov_y[i],
                aspect=cam_aspect[i],
                scale=fs,
                image=thumbnail,
                wxyz=cam_wxyz[i],
                position=cam_positions[i],
                color=(255, 80, 30) if is_current else (60, 60, 60),
                line_width=3.0 if is_current else 1.0,
            )

    print(f"\nViewer running at: http://localhost:{server.get_port()}")
    print("Press Ctrl+C to exit.")

    # Playback loop
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_fps.value)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D point cloud from video using Depth Anything 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Input MP4 video path (required unless --view is used)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PLY file path (required unless --view is used)",
    )
    parser.add_argument(
        "--model", type=str, default="depth-anything/DA3-LARGE-1.1",
        help="HuggingFace model ID for DA3",
    )
    parser.add_argument(
        "--max_frames", type=int, default=30,
        help="Maximum number of frames to extract from video",
    )
    parser.add_argument(
        "--frame_step", type=int, default=1,
        help="Extract every Nth frame",
    )
    parser.add_argument(
        "--native_res", type=int, nargs=2, metavar=("H", "W"), default=None,
        help="Native resolution (H W) to restore aspect ratio before DA3 inference. "
             "Rollout videos are stretched to 256x256; this undoes the distortion. "
             "E.g., --native_res 150 280 for CSGO.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Inference device",
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=40.0,
        help="Confidence percentile for point filtering (0-100)",
    )
    parser.add_argument(
        "--max_points", type=int, default=1_000_000,
        help="Maximum number of points in output PLY",
    )
    parser.add_argument(
        "--process_res", type=int, default=504,
        help="DA3 internal processing resolution",
    )
    parser.add_argument(
        "--save_depth_vis", action="store_true",
        help="Save depth map visualization PNGs",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Launch viser 3D viewer after inference",
    )
    parser.add_argument(
        "--view", type=str, default=None,
        help="View existing data in viser without running inference. "
             "Pass a .npz file for the rich viewer (timeline, depth maps, cameras), "
             "or a .ply file for a simple point cloud viewer.",
    )
    args = parser.parse_args()

    # View-only mode: load data and launch viewer, skip inference
    if args.view is not None:
        view_path = Path(args.view)
        if view_path.suffix == ".npz":
            # Rich viewer from saved viewer data
            viewer_data = load_viewer_data(args.view)
            conf_thresh = viewer_data["conf_threshold"]
            num_frames = len(viewer_data["depth"])
            print(f"Unprojecting {num_frames} frames (conf_threshold={conf_thresh})...")
            per_frame_points = []
            per_frame_colors = []
            for i in range(num_frames):
                pts, cols = unproject_frame(
                    viewer_data["depth"][i],
                    viewer_data["conf"][i],
                    viewer_data["intrinsics"][i],
                    viewer_data["extrinsics"][i],
                    viewer_data["processed_images"][i],
                    conf_thresh,
                )
                per_frame_points.append(pts)
                per_frame_colors.append(cols)
            launch_viewer(per_frame_points, per_frame_colors, viewer_data)
        else:
            # Simple viewer from PLY
            points, colors = load_ply(args.view)
            launch_viewer_ply(points, colors)
        return

    # Validate input for inference mode
    if args.video is None:
        parser.error("--video is required (unless using --view)")
    if args.output is None:
        parser.error("--output is required (unless using --view)")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = Path(args.output)

    # Extract frames
    frames = extract_frames(str(video_path), args.max_frames, args.frame_step)

    # Restore native aspect ratio if needed (rollout videos are stretched to 256x256)
    if args.native_res is not None:
        native_h, native_w = args.native_res
        frames = restore_native_resolution(frames, native_h, native_w)

    # Load model and run inference
    model = load_da3_model(args.model, args.device)
    export_dir = str(output_path.parent / f"{output_path.stem}_da3_export")
    prediction = run_inference(
        model, frames, export_dir, args.process_res,
        args.conf_threshold, args.max_points,
    )

    # Unproject depth maps to 3D point cloud
    print("Unprojecting depth maps to 3D point cloud...")
    per_frame_points = []
    per_frame_colors = []
    for i in range(len(frames)):
        pts, cols = unproject_frame(
            prediction.depth[i],
            prediction.conf[i],
            prediction.intrinsics[i],
            prediction.extrinsics[i],
            prediction.processed_images[i],
            args.conf_threshold,
        )
        per_frame_points.append(pts)
        per_frame_colors.append(cols)
        print(f"  Frame {i}: {len(pts):,} points")

    all_points = np.concatenate(per_frame_points, axis=0)
    all_colors = np.concatenate(per_frame_colors, axis=0)
    print(f"Total points before downsampling: {len(all_points):,}")

    # Downsample if exceeding max_points
    if len(all_points) > args.max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_points), size=args.max_points, replace=False)
        all_points = all_points[indices]
        all_colors = all_colors[indices]
        print(f"Downsampled to {args.max_points:,} points")

    # Save PLY
    save_ply(str(output_path), all_points, all_colors)

    # Save viewer data (.npz) for offline viewing
    viewer_npz_path = str(output_path.parent / f"{output_path.stem}_viewer.npz")
    save_viewer_data(viewer_npz_path, prediction, args.conf_threshold)

    # Save depth visualizations
    if args.save_depth_vis:
        depth_vis_dir = str(output_path.parent / f"{output_path.stem}_depth_vis")
        save_depth_visualizations(prediction, frames, depth_vis_dir)

    # Launch interactive viewer
    if args.visualize:
        viewer_data = {
            "depth": prediction.depth,
            "conf": prediction.conf,
            "intrinsics": prediction.intrinsics,
            "extrinsics": prediction.extrinsics,
            "processed_images": prediction.processed_images,
        }
        launch_viewer(per_frame_points, per_frame_colors, viewer_data)

    print("Done!")


if __name__ == "__main__":
    main()
