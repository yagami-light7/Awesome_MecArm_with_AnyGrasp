import argparse
import sys

import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def rs_intrinsics_to_open3d(intrinsics: rs.intrinsics) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.ppx,
        cy=intrinsics.ppy,
    )


def filter_point_cloud(
    pcd: o3d.geometry.PointCloud,
    depth_min: float,
    depth_max: float,
    voxel_size: float,
    nb_neighbors: int,
    std_ratio: float,
) -> tuple[o3d.geometry.PointCloud, int]:
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(points) == 0:
        return pcd, 0

    valid_mask = np.isfinite(points).all(axis=1)
    valid_mask &= points[:, 2] >= depth_min
    valid_mask &= points[:, 2] <= depth_max

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(points[valid_mask])
    if len(colors) == len(points):
        filtered.colors = o3d.utility.Vector3dVector(colors[valid_mask])

    removed = int(len(points) - len(filtered.points))

    if voxel_size > 0 and len(filtered.points) > 0:
        filtered = filtered.voxel_down_sample(voxel_size)

    if len(filtered.points) > nb_neighbors:
        filtered, _ = filtered.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

    return filtered, removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one RealSense RGB-D frame and visualize it as an Open3D point cloud.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30, help="Number of frames to skip before capture.")
    parser.add_argument("--depth-min", type=float, default=0.15, help="Min depth in meters to keep.")
    parser.add_argument("--depth-max", type=float, default=1.5, help="Max depth in meters to keep.")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Voxel size in meters for downsampling.")
    parser.add_argument("--nb-neighbors", type=int, default=20, help="Neighbors for statistical outlier removal.")
    parser.add_argument("--std-ratio", type=float, default=2.0, help="Std ratio for statistical outlier removal.")
    parser.add_argument("--save", type=str, default="", help="Optional output .ply path.")
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        for _ in range(args.warmup):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("failed to get aligned color/depth frames", file=sys.stderr)
            return 1

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        o3d_color = o3d.geometry.Image(color_image)
        o3d_depth = o3d.geometry.Image(depth_image)
        intrinsic = rs_intrinsics_to_open3d(intrinsics)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0 / depth_scale,
            depth_trunc=args.depth_max,
            convert_rgb_to_intensity=False,
        )
        raw_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd, removed = filter_point_cloud(
            raw_pcd,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            voxel_size=args.voxel_size,
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
        )

        # Match Open3D's conventional camera-facing coordinate system.
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        print(f"raw points: {len(raw_pcd.points)}")
        print(f"filtered points: {len(pcd.points)}")
        print(f"removed by depth/validity filter: {removed}")
        print(f"depth_scale: {depth_scale}")
        if args.save:
            ok = o3d.io.write_point_cloud(args.save, pcd)
            print(f"saved: {args.save} ({'ok' if ok else 'failed'})")

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, frame], window_name="RealSense Point Cloud")
    except Exception as exc:
        print(f"point cloud visualization failed: {exc}", file=sys.stderr)
        return 1
    finally:
        pipeline.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
