import pandas as pd
import numpy as np
import os
from pathlib import Path

class FrontCameraSimulation:
    """Class representing a front camera simulation."""
    def __init__(self, from_id=100, frames=2000):
        self.from_id = from_id
        self.frames = frames

    def generate_data(self):
        # setup data structure
        self.data = pd.DataFrame(columns=["Timestamp", "FrameID", "Speed", "YawRate", "Signal1", "Signal2"])

        # initial values
        self.data.loc[0] = [100000000.0, self.from_id, 60.0, 0.0, 0, 0.0]

        # generate data by frame
        for row_index in range(1, self.frames):
            # copy last row 
            new_row = self.data.loc[row_index-1].copy()

            # setup incremental values
            time_increment = 1000 * (27.7 + np.random.uniform(-0.05,  0.05)) # ms to microseconds conversion
            speed_increment = 0.08

            # calculate frame ID
            frame_id = self.from_id + row_index

            # update values
            new_row["Timestamp"] += time_increment
            new_row["FrameID"] = frame_id

            if new_row["Speed"] + speed_increment <= 120:
                new_row["Speed"] += speed_increment
            else:
                new_row["Speed"] = 120.0 + np.random.uniform(-0.05, 0.05)

            new_row["YawRate"] = np.random.uniform(-1.0, 1.0)
            
            # set signal 1 value only once
            if frame_id == 201:
                new_row["Signal1"] = np.random.randint(1, 16)

            # update signal 2 based on signal 1
            if new_row["Signal1"] >= 5:
                new_row["Signal2"] = 80 + np.random.uniform(-10.0, 10.0)

            # add the new row to the DataFrame
            self.data.loc[row_index] = new_row
        return self.data
    
    def format_data(self):
        # format data for output csv
        self.data["FrameID"] = self.data["FrameID"].astype(int)
        self.data["Signal1"] = self.data["Signal1"].astype(int)

        self.data['Timestamp'] = self.data['Timestamp'].apply(lambda x: f"{x:.6f}")
        self.data['Speed'] = self.data['Speed'].apply(lambda x: f"{x:.2f}")
        self.data['YawRate'] = self.data['YawRate'].apply(lambda x: f"{x:.2f}")
        self.data['Signal2'] = self.data['Signal2'].apply(lambda x: f"{x:.2f}")

        return self.data

    def to_csv(self, output_dir):
        self.generate_data()
        self.format_data()

        # create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # save to CSV in defined location
        output_path = os.path.join(output_dir, "f_cam_out.csv")
        self.data.to_csv(output_path, index=False)
        
# when directly running the script
if __name__ == "__main__":
    # for command line output path definition
    import argparse
    parser = argparse.ArgumentParser(description="Generates front camera simulation data and saves it to CSV.")
    parser.add_argument("--output_dir", type=str, default="data", help="output directory path")
    args = parser.parse_args()

    # create simulation and save to CSV
    simulation = FrontCameraSimulation()
    simulation.to_csv(args.output_dir)