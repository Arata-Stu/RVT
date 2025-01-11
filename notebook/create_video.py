import sys
sys.path.append('..')
import argparse
from omegaconf import OmegaConf
from visualize import VideoWriter


from config.modifier import dynamically_modify_train_config

def main(yaml_path: str, output_path: str, fps: int=10, max_sequence_length: int=1):
    video_writer = VideoWriter(output_path, fps)

    config = OmegaConf.load(yaml_path)
    dynamically_modify_train_config(config)
    video_writer.run(config, max_sequence_length)
    video_writer.video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-y','--yaml_path', type=str, required=True)
    parser.add_argument('-o','--output_path', type=str, required=True)
    parser.add_argument('-f','--fps', type=int, default=10)
    parser.add_argument('-m','--max_sequence_length', type=int, default=1)

    args = parser.parse_args()
    main(args.yaml_path, args.output_path, args.fps, args.max_sequence_length)
