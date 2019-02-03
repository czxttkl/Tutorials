import argparse
import sys

parser = argparse.ArgumentParser(
        description="Train a RL net to play in an OpenAI Gym environment.",
        allow_abbrev=False
    )
parser.add_argument("-p", "--parameters", help="Path to JSON parameters file.")
parser.add_argument("--offline_train", action='store_true')
raw_args = sys.argv[1:]
args = parser.parse_args(raw_args)
print(raw_args)
print(args)



