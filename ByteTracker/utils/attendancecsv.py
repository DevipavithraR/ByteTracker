import csv
from datetime import datetime

def mark_attendance(name):
    """Append attendance record to CSV file."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])
