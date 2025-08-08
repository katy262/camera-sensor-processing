import pandas as pd
import numpy as np
import os
from pathlib import Path

class Reprocessor:
    def __init__(self, camera_data_path, sensor_data_path):
        self.sensor_data_path = sensor_data_path
        self.camera_data_path = camera_data_path
        self.data = None
    
    def try_load_data(self):
        if not os.path.exists(self.sensor_data_path) or not os.path.exists(self.camera_data_path):
            print(f"Error: One or both input files do not exist: {self.sensor_data_path}, {self.camera_data_path}")
            print("Please check the file paths and try again.")
            return False

        self.sensor_data = pd.read_csv(self.sensor_data_path)
        self.camera_data = pd.read_csv(self.camera_data_path)
        return True

    def reprocess_data(self):
        # load data for processing
        if not self.try_load_data():
            # return None if loading fails
            return None
        
        # copy camera data
        self.data = self.camera_data.copy()

        sensor_index = 0
        # iterate over each row
        for index, row in self.camera_data.iterrows():
            # find last sensor timestamp
            while (sensor_index + 1 < len(self.sensor_data) and 
                   self.sensor_data.loc[sensor_index + 1, "Timestamp"] <= row["Timestamp"]):
                    sensor_index += 1
            # average speed with sensor data
            self.data.loc[index, "Speed"] = (row["Speed"] + self.sensor_data.at[sensor_index, "Speed"])/2
        
        return self.data
    
    def format_data(self):
        self.data["FrameID"] = self.data["FrameID"].astype(int)
        self.data["Signal1"] = self.data["Signal1"].astype(int)

        self.data['Timestamp'] = self.data['Timestamp'].apply(lambda x: f"{x:.6f}")
        self.data['Speed'] = self.data['Speed'].apply(lambda x: f"{x:.2f}")
        self.data['YawRate'] = self.data['YawRate'].apply(lambda x: f"{x:.2f}")
        self.data['Signal2'] = self.data['Signal2'].apply(lambda x: f"{x:.2f}")

        return self.data

    def to_csv(self, output_dir):
        if self.reprocess_data() is not None:
            self.format_data()

            # create directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # save to CSV in defined location
            output_path = os.path.join(output_dir, "resim_out.csv")
            self.data.to_csv(output_path, index=False)
        
# when directly running the script
if __name__ == "__main__":
    # for command line inputs location definition and output path definition
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_sensor", type=str, default="data/sensor_out.csv", help="input sensor CSV file path")
    parser.add_argument("--input_camera", type=str, default="data/f_cam_out.csv", help="input front camera CSV file path")
    parser.add_argument("--output_dir", type=str, default="data", help="output directory path")
    args = parser.parse_args()

    # process data and save to CSV
    processing = Reprocessor(args.input_camera, args.input_sensor)
    processing.to_csv(args.output_dir)


