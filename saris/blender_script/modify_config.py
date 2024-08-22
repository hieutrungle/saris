import yaml
import argparse
import os
import re


def main():
    args = create_argparser().parse_args()
    with open(args.config_file) as istream:
        ymldoc = yaml.safe_load(istream)
        ymldoc["rx_position"] = args.rx_position
        if re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", args.num_samples):
            ymldoc["cm_num_samples"] = int(float(args.num_samples))
        elif args.num_samples.isnumeric():
            ymldoc["cm_num_samples"] = int(args.num_samples)
        else:
            raise ValueError("num_samples must be a number")

    # add tmp to args.config_file
    tmp_file = args.config_file.split(".")[0] + "_tmp.yaml"
    with open(tmp_file, "w") as ostream:
        yaml.dump(ymldoc, ostream, default_flow_style=False, sort_keys=False)

    # move/overwrite tmp file to output file
    os.rename(tmp_file, args.output)


def create_argparser() -> argparse.ArgumentParser:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument(
        "--rx_position",
        "-rx_pos",
        nargs="*",  # 0 or more values expected => creates a list
        type=float,
        default=[0, 0, 0],  # default if nothing is provided
        required=True,
    )
    parser.add_argument("--num_samples", "-ns", type=str, default="1e6")
    return parser


if __name__ == "__main__":
    main()
