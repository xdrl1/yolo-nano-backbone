import requests
import time
import os

# Tokens
user_key = "unfbsvg8dktbcvcdfdfoqj12j79npj"
api_token = "asjj9knhiebwxt1uin9cuo4o44cypi"

# Path to log file
log_file = "../YOLOX_outputs/yolox_nano_deploy_relu_bird/train_log.txt"

# Function to send notification
def send_notification(epoch):
    message = f"Started training epoch {epoch}"
    r = requests.post("https://api.pushover.net/1/messages.json", data={
        "token": api_token,
        "user": user_key,
        "message": message
    })
    print(f"Notification sent for epoch {epoch}: {r.text}")

# Find the size of the log file and go to the end
file_end = os.path.getsize(log_file)

# Monitor the log file
last_epoch = 0

while True:
    try:
        # Check if the log file has been updated
        current_end = os.path.getsize(log_file)
        if current_end > file_end:
            with open(log_file, "rb") as f:
                # Go to the last known end of the file
                f.seek(file_end)
                # Read new content
                new_content = f.read().decode('utf-8')
                file_end = current_end  # Update the last known end

            # Split new content into lines and process each line
            for line in new_content.splitlines():
                # Look for the phrase that precedes the epoch number
                marker = "---> start train epoch"
                if marker in line:
                    try:
                        # Extract the epoch number, which comes after the marker
                        epoch_str = line.split(marker)[-1].strip()
                        current_epoch = int(epoch_str)

                        if current_epoch > last_epoch:
                            send_notification(current_epoch)
                            last_epoch = current_epoch
                            if current_epoch >= 160:  # Terminate after epoch 160
                                print("Reached epoch 150. Exiting script.")
                                exit()
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing epoch from line: {line}")
                        print(str(e))
                        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    time.sleep(60)  # Sleep for 60 seconds before checking the file again
