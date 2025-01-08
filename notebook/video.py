import sys
sys.path.append('..')

from visualize import create_video_from_dataloader

yaml_path = "./config/dt_50.yaml"
create_video_from_dataloader(yaml_path, output_path="output_video.mp4", fps=10, max_sequences=1)