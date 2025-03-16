# monitor.py
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

class DataChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith((".csv")): #only run if csv files are modified.
            print(f"File {event.src_path} has been modified. Running project...")
            subprocess.run(["python", "nifty_options_simulator.py"])

if __name__ == "__main__":
    path = os.path.join("..//data") #monitor the data folder.
    event_handler = DataChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()