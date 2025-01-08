import sys
sys.path.append('..')

from visualize import visualize_detection

yaml_path = "./config/dt_50.yaml"
visualize_detection(yaml_path, output_path="output_video.mp4", fps=10, max_sequences=1)