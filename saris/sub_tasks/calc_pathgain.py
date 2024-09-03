import os
import argparse

gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from saris.utils import utils
from saris.sigmap import signal_cmap
import tensorflow as tf
import pickle


def main():
    args = create_args()
    config = utils.load_config(args.config_file)

    if args.verbose:
        utils.log_args(args)
        utils.log_config(config)

    tf.random.set_seed(args.seed)

    # Prepare folders
    sig_cmap = signal_cmap.SignalCoverageMap(
        config, args.compute_scene_path, args.viz_scene_path, args.verbose
    )

    if not args.use_cmap:
        paths = sig_cmap.compute_paths()
        a, _ = paths.cir()
        a = tf.squeeze(a)
        path_gain = tf.reduce_mean(tf.reduce_sum(tf.abs(a) ** 2, axis=-1)).numpy()

    else:
        coverage_map = sig_cmap.compute_cmap()
        path_gain = sig_cmap.get_path_gain(coverage_map)
        sig_cmap.render_to_file(coverage_map, None, filename=args.saved_path)

    results_dict = {"path_gain": path_gain}
    with open(args.results_path, "wb") as f:
        pickle.dump(results_dict, f)


def create_args() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--compute_scene_path", "-cp", type=str, required=True)
    parser.add_argument("--viz_scene_path", "-vp", type=str)
    parser.add_argument("--saved_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--use_cmap", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.viz_scene_path is None or args.viz_scene_path == "":
        args.viz_scene_path = args.compute_scene_path
    return args


if __name__ == "__main__":
    main()
