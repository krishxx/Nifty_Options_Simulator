import pandas as pd
import time
import os
from datetime import datetime

def append_data_with_sleep(source_file, destination_file, processed_rows):
    """
    Appends one unprocessed row from a source CSV to a destination CSV with a 1-second sleep.
    """

    try:
        source_df = pd.read_csv(source_file)
        if source_df.empty:
            print(f"Source file {source_file} is empty. No data to append.")
            return processed_rows

        for index, row in source_df.iterrows():
            if index not in processed_rows:
                # Append the row to the destination file
                if os.path.exists(destination_file):
                    destination_df = pd.read_csv(destination_file)
                    updated_df = pd.concat([destination_df, pd.DataFrame([row])], ignore_index=True)
                    updated_df.to_csv(destination_file, index=False)
                else:
                    pd.DataFrame([row]).to_csv(destination_file, index=False) #create the file with the first row.

                print(f"Appended row {index} from {source_file} to {destination_file} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                processed_rows.add(index)
                time.sleep(1)  # Sleep for 1 second
                return processed_rows #return after one row has been processed.
        return processed_rows

    except FileNotFoundError:
        print(f"Error: File not found. Check the paths: {source_file} or {destination_file}")
        return processed_rows
    except Exception as e:
        print(f"An error occurred: {e}")
        return processed_rows

if __name__ == "__main__":
    sources_destinations = [
        (os.path.join("..", "data", "gendata", "big_oi_summary_rev22025-03-12.csv"), os.path.join("..", "data", "big_oi_summary_rev.csv")),
        (os.path.join("..", "data", "gendata", "cum_sent_df2025-03-12.csv"), os.path.join("..", "data", "cum_sent_df.csv")),
        (os.path.join("..", "data", "gendata", "nifty_fut_12Mar2025.csv"), os.path.join("..", "data", "nifty_fut.csv")),
        (os.path.join("..", "data", "gendata", "nifty_spot_12Mar2025.csv"), os.path.join("..", "data", "nifty_spot.csv")),
    ]

    processed_rows_dict = {source: set() for source, _ in sources_destinations} #dict to store processed row index.

    while True:
        for source, destination in sources_destinations:
            processed_rows_dict[source] = append_data_with_sleep(source, destination, processed_rows_dict[source])
        time.sleep(3)