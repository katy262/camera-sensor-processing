import pandas as pd
import numpy as np  

class Reprocessor:
    def __init__(self, sensor_data_path, camera_data_path):
        self.sensor_data_path = sensor_data_path
        self.camera_data_path = camera_data_path
    
    def load_data(self):
        self.sensor_data = pd.read_csv(self.sensor_data_path)
        self.camera_data = pd.read_csv(self.camera_data_path)

    def reprocess_data(self):
        # load data for processing
        self.load_data()
        # copy camera data
        self.data = self.camera_data.copy()

        sensor_index = 0
        # iterate over each row
        for index, row in self.data.iterrows():
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

    def to_csv(self, file_path):
        self.reprocess_data()
        self.format_data()
        self.data.to_csv(file_path, index=False)
        
# when directly running the script
if __name__ == "__main__":
    # for command line inputs location definition and output path definition
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sensor", type=str, default="data/sensor_out.csv", help="input sensor CSV file path")
    parser.add_argument("--input-camera", type=str, default="data/f_cam_out.csv", help="input front camera CSV file path")
    parser.add_argument("--output", type=str, default="data/resim_out.csv", help="output CSV file path")
    args = parser.parse_args()

    # process data and save to CSV
    simulation = Reprocessor(args.input_sensor, args.input_camera)
    simulation.to_csv(args.output)


