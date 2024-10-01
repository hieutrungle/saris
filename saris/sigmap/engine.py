from saris.utils import utils, timer
import sionna.rt
import gc
from typing import Optional
import os
import tensorflow as tf
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera


def prepare_scene(config, filename, cam=None):
    # Scene Setup
    scene = load_scene(filename)

    # in Hz; implicitly updates RadioMaterials
    scene.frequency = config["frequency"]
    # If set to False, ray tracing will be done per antenna element (slower for large arrays)
    scene.synthetic_array = config["synthetic_array"]

    if cam is not None:
        scene.add(cam)

    # Device Setup
    scene.tx_array = PlanarArray(
        num_rows=config["tx_num_rows"],
        num_cols=config["tx_num_cols"],
        vertical_spacing=config["tx_vertical_spacing"],
        horizontal_spacing=config["tx_horizontal_spacing"],
        pattern=config["tx_pattern"],
        polarization=config["tx_polarization"],
    )
    for i, (tx_pos, tx_orient) in enumerate(zip(config["tx_positions"], config["tx_orientations"])):
        tx = Transmitter(f"tx_{i}", tx_pos, tx_orient)
        scene.add(tx)

    scene.rx_array = PlanarArray(
        num_rows=config["rx_num_rows"],
        num_cols=config["rx_num_cols"],
        vertical_spacing=config["rx_vertical_spacing"],
        horizontal_spacing=config["rx_horizontal_spacing"],
        pattern=config["rx_pattern"],
        polarization=config["rx_polarization"],
    )

    for i, (rx_pos, rx_orient) in enumerate(zip(config["rx_positions"], config["rx_orientations"])):
        rx = Receiver(f"rx_{i}", rx_pos, rx_orient)
        scene.add(rx)

    return scene


def prepare_camera(config):
    cam = Camera(
        "my_cam",
        position=config["cam_position"],
        orientation=config["cam_orientation"],
    )
    cam.look_at(config["cam_look_at"])
    return cam


class SignalCoverageMap:
    def __init__(
        self,
        config,
        compute_scene_path: str,
        viz_scene_path: str,
        verbose: bool = False,
    ):

        self.config = config

        # input directories
        self._compute_scene_path = compute_scene_path
        self._viz_scene_path = viz_scene_path

        # Camera
        self._cam = prepare_camera(self.config)

        self.verbose = verbose

    @property
    def cam(self):
        return self._cam

    def compute_cmap(self, **kwargs) -> sionna.rt.CoverageMap:
        # Compute coverage maps with ceiling on
        if self.verbose:
            print(f"Computing coverage map for {self._compute_scene_path}")
            with timer.Timer(
                text="Elapsed coverage map time: {:0.4f} seconds\n",
                print_fn=print,
            ):
                cmap = self._compute_cmap(**kwargs)
        else:
            cmap = self._compute_cmap(**kwargs)

        self.free_memory()
        return cmap

    def _compute_cmap(self, **kwargs) -> sionna.rt.CoverageMap:
        scene = prepare_scene(self.config, self._compute_scene_path, self.cam)

        cm_kwargs = dict(
            max_depth=self.config["cm_max_depth"],
            cm_cell_size=self.config["cm_cell_size"],
            num_samples=self.config["cm_num_samples"],
            diffraction=self.config["diffraction"],
        )
        if kwargs:
            cm_kwargs.update(kwargs)

        cmap = scene.coverage_map(**cm_kwargs)
        return cmap

    def compute_paths(self, **kwargs) -> sionna.rt.Paths:
        # Compute coverage maps with ceiling on
        if self.verbose:
            print(f"Computing paths for {self._compute_scene_path}")
            with timer.Timer(
                text="Elapsed paths time: {:0.4f} seconds\n",
                print_fn=print,
            ):
                paths = self._compute_paths(**kwargs)
        else:
            paths = self._compute_paths(**kwargs)

        self.free_memory()
        return paths

    def _compute_paths(self, **kwargs) -> sionna.rt.Paths:
        scene = prepare_scene(self.config, self._compute_scene_path, self.cam)

        paths_kwargs = dict(
            max_depth=self.config["path_max_depth"],
            num_samples=self.config["path_num_samples"],
        )
        if kwargs:
            paths_kwargs.update(kwargs)

        paths = scene.compute_paths(**paths_kwargs)
        return paths

    def compute_render(self, cmap_enabled: bool = False, paths_enabled: bool = False) -> None:

        # Visualize coverage maps with ceiling off
        cm = self.compute_cmap() if cmap_enabled else None
        paths = self.compute_paths() if paths_enabled else None

        scene = prepare_scene(self.config, self._viz_scene_path, self.cam)

        img_dir = utils.get_image_dir(self.config)
        render_filename = utils.create_filename(
            img_dir, f"{self.config['mitsuba_filename']}_00000.png"
        )
        render_config = dict(
            camera=self.cam,
            paths=paths,
            filename=render_filename,
            coverage_map=cm,
            cm_vmin=self.config["cm_vmin"],
            cm_vmax=self.config["cm_vmax"],
            resolution=self.config["resolution"],
            show_devices=True,
        )
        scene.render_to_file(**render_config)

    def render_to_file(
        self,
        coverage_map: sionna.rt.CoverageMap = None,
        paths: sionna.rt.Paths = None,
        filename: Optional[str] = None,
    ) -> None:
        scene = prepare_scene(self.config, self._viz_scene_path, self.cam)

        if filename is None:
            img_dir = utils.get_image_dir(self.config)
            render_filename = utils.create_filename(
                img_dir, f"{self.config['mitsuba_filename']}_00000.png"
            )
        else:
            render_filename = filename
        render_config = dict(
            camera=self.cam,
            paths=paths,
            filename=render_filename,
            coverage_map=coverage_map,
            cm_vmin=self.config["cm_vmin"],
            cm_vmax=self.config["cm_vmax"],
            resolution=self.config["resolution"],
            show_devices=True,
        )
        scene.render_to_file(**render_config)

    def render(
        self, coverage_map: sionna.rt.CoverageMap = None, paths: sionna.rt.Paths = None
    ) -> sionna.rt.Scene:
        scene = prepare_scene(self.config, self._viz_scene_path, self.cam)
        render_config = dict(
            camera=self.cam,
            paths=paths,
            coverage_map=coverage_map,
            cm_vmin=self.config["cm_vmin"],
            cm_vmax=self.config["cm_vmax"],
            resolution=self.config["resolution"],
            show_devices=True,
        )
        scene.render(**render_config)
        return scene

    def get_path_gain_slow(self, coverage_map: sionna.rt.CoverageMap) -> float:

        coverage_map_tensor = coverage_map.as_tensor()
        coverage_map_centers = coverage_map.cell_centers
        rx_position = self.config["rx_position"]
        distances = tf.norm(coverage_map_centers - rx_position, axis=-1)
        min_dist = tf.reduce_min(distances)
        min_ind = tf.where(tf.equal(distances, min_dist))[0]

        path_gain: tf.Tensor = coverage_map_tensor[0, min_ind[0], min_ind[1]]
        path_gain = float(path_gain.numpy())
        return path_gain

    def get_path_gain(self, coverage_map: sionna.rt.CoverageMap) -> float:
        coverage_map_tensor = coverage_map.as_tensor()
        coverage_map_centers = coverage_map.cell_centers
        rx_positions = tf.convert_to_tensor(self.config["rx_positions"], dtype=tf.float32)
        # Get the top left position of the coverage map
        top_left_pos = coverage_map_centers[0, 0, 0:2] - (coverage_map.cell_size / 2)
        path_gains = []
        for rx_position in rx_positions:
            x_distance_to_top_left = rx_position[0] - top_left_pos[0]
            y_distance_to_top_left = rx_position[1] - top_left_pos[1]

            ind_y = int(y_distance_to_top_left / coverage_map.cell_size[1])
            ind_x = int(x_distance_to_top_left / coverage_map.cell_size[0])

            path_gain: tf.Tensor = coverage_map_tensor[0, ind_y, ind_x]
            path_gain = float(path_gain.numpy())
            path_gains.append(path_gain)
        return path_gains

    def get_viz_scene(self) -> sionna.rt.Scene:
        scene = prepare_scene(self.config, self._viz_scene_path, self.cam)
        return scene

    def free_memory(self) -> None:
        tf.keras.backend.clear_session()
        gc.collect()

        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            tf.config.experimental.reset_memory_stats("GPU:0")
