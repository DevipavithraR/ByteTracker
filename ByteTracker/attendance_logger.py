# attendance_logger.py
import csv
from datetime import datetime

CSV_FILE = "attendance.csv"

def ensure_csv_exists():
    """Create CSV with headers if not exists."""
    try:
        with open(CSV_FILE, "x") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "employee_name", "login_time", "status", "logout_time"])
    except FileExistsError:
        pass


def mark_login(employee_name, login_time):
    """Add login entry if not already present for today."""
    ensure_csv_exists()
    today = login_time.date()

    # Prevent duplicate login entries
    rows = []
    exists = False
    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in rows:
        if row[0] == today.strftime("%Y-%m-%d") and row[1] == employee_name:
            exists = True
            break

    if exists:
        return

    # Decide status: Present or Late
    lt = login_time.time()

    status = "Present"
    if lt >= datetime.strptime("10:00", "%H:%M").time():
        status = "Late"

    with open(CSV_FILE, "a") as f:
        writer = csv.writer(f)
        writer.writerow([
            today.strftime("%Y-%m-%d"),
            employee_name,
            login_time.strftime("%H:%M:%S"),
            status,
            ""  # logout empty now
        ])


def mark_logout(employee_name, logout_time):
    """Update logout time for today's row."""
    ensure_csv_exists()
    
    today_str = logout_time.strftime("%Y-%m-%d")
    rows = []

    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in rows:
        if len(row) < 5:
            continue
        if row[0] == today_str and row[1] == employee_name:
            row[4] = logout_time.strftime("%H:%M:%S")
            break

    with open(CSV_FILE, "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
