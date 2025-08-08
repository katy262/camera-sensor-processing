import argparse
import os
from pathlib import Path

from f_cam import FrontCameraSimulation
from sensor import SensorSimulation
from resim import Reprocessor

def main():
    # define command line arguments for input and output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data", help="output CSV file directory")
    parser.add_argument("--input_dir", type=str, default="data", help="input CSV files directory")
    args = parser.parse_args()

    # simulate front camera 
    fc_simulation = FrontCameraSimulation()
    fc_simulation.to_csv(args.input_dir)

    # simulate sensor
    sensor_simulation = SensorSimulation()
    sensor_simulation.to_csv(args.input_dir)

    # reprocess data
    reprocessor = Reprocessor(
        sensor_data_path=os.path.join(args.input_dir, "sensor_out.csv"),
        camera_data_path=os.path.join(args.input_dir, "f_cam_out.csv")
    )
    # save reprocessed data to CSV
    reprocessor.to_csv(args.output_dir)

# running the script
if __name__ == "__main__":
    main()