from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera


def prepare_scene(config, filename, cam=None):
    # Scene Setup
    scene = load_scene(filename)

    # in Hz; implicitly updates RadioMaterials
    scene.frequency = config.frequency
    # If set to False, ray tracing will be done per antenna element (slower for large arrays)
    scene.synthetic_array = config.synthetic_array

    if cam is not None:
        scene.add(cam)

    # Device Setup
    scene.tx_array = PlanarArray(
        num_rows=config.tx_num_rows,
        num_cols=config.tx_num_cols,
        vertical_spacing=config.tx_vertical_spacing,
        horizontal_spacing=config.tx_horizontal_spacing,
        pattern=config.tx_pattern,
        polarization=config.tx_polarization,
    )
    tx = Transmitter("tx", config.tx_position, config.tx_orientation)
    scene.add(tx)

    scene.rx_array = PlanarArray(
        num_rows=config.rx_num_rows,
        num_cols=config.rx_num_cols,
        vertical_spacing=config.rx_vertical_spacing,
        horizontal_spacing=config.rx_horizontal_spacing,
        pattern=config.rx_pattern,
        polarization=config.rx_polarization,
    )
    if config.rx_included:
        rx = Receiver("rx", config.rx_position, config.rx_orientation)
        scene.add(rx)

    return scene


def prepare_camera(config):
    cam = Camera(
        "my_cam",
        position=config.cam_position,
        orientation=config.cam_orientation,
    )
    cam.look_at(config.cam_look_at)
    return cam
