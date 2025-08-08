import argparse
import os
from pathlib import Path

from f_cam import FrontCameraSimulation
from sensor import SensorSimulation
from resim import Reprocessor

"""Main script to run the camera and sensor simulations, and process the data."""
def main():
    # define command line arguments for output files directory
    parser = argparse.ArgumentParser(description="Creates simulation data and processes it.")
    parser.add_argument("--output_dir", type=str, default="data", help="output directory path")
    args = parser.parse_args()

    # simulate front camera 
    fc_simulation = FrontCameraSimulation()
    fc_simulation.to_csv(args.output_dir)
    print(f"Front camera simulation data saved to csv file {args.output_dir}/f_cam_out.csv")

    # simulate sensor
    sensor_simulation = SensorSimulation()
    sensor_simulation.to_csv(args.output_dir)
    print(f"Sensor simulation data saved to csv file {args.output_dir}/sensor_out.csv")

    # reprocess data
    print("Processing simulated data...")
    reprocessor = Reprocessor(
        sensor_data_path=os.path.join(args.output_dir, "sensor_out.csv"),
        camera_data_path=os.path.join(args.output_dir, "f_cam_out.csv")
    )
    # save reprocessed data to CSV
    reprocessor.to_csv(args.output_dir)

# running the script
if __name__ == "__main__":
    main()