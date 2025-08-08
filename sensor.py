import pandas as pd
import numpy as np

class SensorSimulation:
    def __init__(self, from_s=100, to_s=160):
        # convert seconds to microseconds
        conversion_factor = 1000000
        self.start = conversion_factor * from_s
        self.end = conversion_factor * to_s

    def generate_data(self):
        # setup data structure
        self.data = pd.DataFrame(columns=["Timestamp", "Speed"])

        # initial values
        self.data.loc[0] = [self.start, 60.0]
        row_index = 0

        # generate data until 160s
        while self.data.loc[row_index]["Timestamp"] < self.end:
            # copy last row 
            new_row = self.data.loc[row_index].copy()

            # setup incremental values
            time_increment = 1000 * (200 + np.random.uniform(-10, 10)) # ms to microseconds conversion
            speed_increment = 0.56

            # update values
            new_row["Timestamp"] += time_increment

            if new_row["Speed"] < 120:
                new_row["Speed"] += speed_increment
            else:
                new_row["Speed"] = 120.0 + np.random.uniform(-0.1, 0.1)

            # throw away the last row if it exceeds the end time
            if new_row["Timestamp"] > self.end:
                break

            # add the new row to the DataFrame
            row_index += 1
            self.data.loc[row_index] = new_row

        return self.data
    
    def format_data(self):
        self.data['Timestamp'] = self.data['Timestamp'].apply(lambda x: f"{x:.6f}")
        self.data['Speed'] = self.data['Speed'].apply(lambda x: f"{x:.2f}")
        return self.data

    def to_csv(self, file_path):
        self.generate_data()
        self.format_data()
        self.data.to_csv(file_path, index=False)
        
# when directly running the script
if __name__ == "__main__":
    # for command line output path definition
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/sensor_out.csv", help="output CSV file path")
    args = parser.parse_args()

    # create simulation and save to CSV
    simulation = SensorSimulation()
    simulation.to_csv(args.output)