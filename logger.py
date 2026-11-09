
import csv
import os
import re
import subprocess
import time
import uuid
from datetime import datetime

DATA_FOLDER = "data"
INTERVAL_SECONDS = 1

def get_wifi_signal():
    """Read Wi-Fi signal percentage using Windows netsh."""
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        match = re.search(
            r"Signal\s*:\s*(\d+)\s*%",
            result.stdout,
            re.IGNORECASE
        )

        if match:
            percentage = int(match.group(1))

            # Approximate conversion, not a measured dBm value.
            estimated_rssi = -100 + (percentage * 0.5)

            return percentage, round(estimated_rssi, 1)

        return None, None

    except Exception as error:
        print("Wi-Fi reading error:", error)
        return None, None


def main():
    print("\n===== Wi-Fi Sensing Data Logger =====")

    room = input("Enter room name (example: Lab1): ").strip()
    if not room:
        room = "Unknown"

    while True:
        try:
            headcount = int(input("Enter number of people in the room: "))
            if headcount < 0:
                print("Headcount cannot be negative.")
                continue
            break
        except ValueError:
            print("Please enter a valid whole number.")

    session_id = str(uuid.uuid4())[:8]
    os.makedirs(DATA_FOLDER, exist_ok=True)

    filename = os.path.join(
        DATA_FOLDER,
        f"{room}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    print("\nRecording started.")
    print("Press Ctrl+C to stop.\n")

    try:
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "signal_percent",
                "rssi_estimated_dbm",
                "room",
                "manual_headcount",
                "session_id"
            ])

            while True:
                timestamp = datetime.now().isoformat(timespec="seconds")
                percentage, rssi = get_wifi_signal()

                if percentage is not None:
                    writer.writerow([
                        timestamp,
                        percentage,
                        rssi,
                        room,
                        headcount,
                        session_id
                    ])
                    file.flush()

                    print(
                        f"{timestamp} | Signal: {percentage}% "
                        f"| Estimated RSSI: {rssi} dBm"
                    )
                else:
                    print("Wi-Fi signal unavailable. Check your connection.")

                time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nRecording stopped.")

    print("CSV saved to:", filename)


if __name__ == "__main__":
    main()
