from saris.utils import utils, timer, map_prep
import os


@timer.Timer(text="Elapsed coverage map time: {:0.4f} seconds\n")
def compute_coverage_map(args, config):
    # Prepare folders
    cm_scene_folders, viz_scene_folders = utils.get_input_folders(args, config)
    img_folder, img_tmp_folder, video_folder = utils.get_output_folders(config)

    # Compute coverage maps
    for i, (cm_scene_folder, viz_scene_folder) in enumerate(
        zip(cm_scene_folders, viz_scene_folders)
    ):
        # Compute coverage maps with ceiling on
        cam = map_prep.prepare_camera(config)
        filename = os.path.join(cm_scene_folder, f"{config.mitsuba_filename}.xml")
        print.log(f"Computing coverage map for {filename}")
        scene = map_prep.prepare_scene(config, filename, cam)

        cm = scene.coverage_map(
            max_depth=config.cm_max_depth,
            cm_cell_size=config.cm_cell_size,
            num_samples=config.cm_num_samples,
            diffraction=config.diffraction,
        )
        # paths = scene.compute_paths(
        #     max_depth=config.path_max_depth, num_samples=config.path_num_samples
        # )
        paths = None

        # Visualize coverage maps
        filename = os.path.join(viz_scene_folder, f"{config.mitsuba_filename}.xml")
        scene = map_prep.prepare_scene(config, filename, cam)

        render_filename = utils.create_filename(
            img_tmp_folder, f"{config.mitsuba_filename}_{i:03d}.png"
        )
        render_config = dict(
            camera=cam,
            paths=paths,
            # filename=os.path.join(
            #     img_tmp_folder, f"{config.mitsuba_filename}_{i:02d}.png"
            # ),
            filename=render_filename,
            coverage_map=cm,
            cm_vmin=config.cm_vmin,
            cm_vmax=config.cm_vmax,
            resolution=config.resolution,
            show_devices=True,
        )
        scene.render_to_file(**render_config)

        if i == 0 and i == len(cm_scene_folders) - 1:
            render_config["filename"] = os.path.join(
                img_folder, f"{config.scene_name}_scene_{i:03d}.png"
            )
            # render_config["show_devices"] = True
            scene.render_to_file(**render_config)
