import os
import argparse

gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sigmap.utils import utils, logger, scripting_utils
from sigmap import compute
import tensorflow as tf
import json
import sionna


def main():
    args = create_args()
    config = scripting_utils.make_sionna_config(args.config_file)
    log_dir = "./tmp_logs"
    utils.mkdir_not_exists(log_dir)
    logger.configure(dir=log_dir)

    logger.log(f"using tensorflow version: {tf.__version__}")
    if tf.config.list_physical_devices("GPU") == []:
        logger.log(f"no GPU available\n")
    else:
        logger.log(f"Available GPUs: {tf.config.list_physical_devices('GPU')}\n")

    if args.verbose:
        utils.log_args(args)
        utils.log_config(config)

    tf.random.set_seed(args.seed)

    # Prepare folders
    sig_cmap = compute.signal_cmap.SignalCoverageMap(
        config, args.compute_scene_path, args.viz_scene_path, args.verbose
    )

    if not args.use_cmap:
        paths = sig_cmap.compute_paths()
        a, tau = paths.cir()
        a = tf.squeeze(a)
        path_gain = tf.reduce_mean(tf.reduce_sum(tf.abs(a) ** 2, axis=-1)).numpy()

    else:
        coverage_map = sig_cmap.compute_cmap()
        path_gain = sig_cmap.get_path_gain(coverage_map)

        sig_cmap.render_to_file(coverage_map, None, filename=args.saved_path)

    results_name = "path_gain-" + args.saved_path.split("/")[-2] + ".txt"
    tmp_dir = utils.get_tmp_dir()
    results_file = os.path.join(tmp_dir, results_name)
    # results_file = os.path.join(tmp_dir, "path_gain.txt")
    results_dict = {
        "path_gain": path_gain,
    }
    with open(results_file, "w") as f:
        json.dump(results_dict, f, cls=utils.NpEncoder)


def create_args() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    defaults = dict()
    # defaults.update(utils.rt_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--compute_scene_path", "-cp", type=str, required=True)
    parser.add_argument("--viz_scene_path", "-vp", type=str)
    parser.add_argument("--saved_path", type=str, default=None)
    parser.add_argument("--use_cmap", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)

    scripting_utils.add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()
    if args.viz_scene_path is None or args.viz_scene_path == "":
        args.viz_scene_path = args.compute_scene_path
    return args


if __name__ == "__main__":
    main()
