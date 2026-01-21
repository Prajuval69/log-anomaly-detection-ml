import re
import pandas as pd

log_file = "HDFS.log"

pattern = re.compile(
    r'(?P<date>\d{6})\s+'
    r'(?P<time>\d{6})\s+'
    r'(?P<pid>\d+)\s+'
    r'(?P<level>\w+)\s+'
    r'(?P<component>[\w\.\$]+):\s+'
    r'(?P<message>.*)'
)

logs = []

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            logs.append(match.groupdict())

df = pd.DataFrame(logs)

# Convert to timestamp (YYMMDD HHMMSS)
df["timestamp"] = pd.to_datetime(df["date"] + df["time"], format="%y%m%d%H%M%S")

df = df.drop(columns=["date", "time"])

df.to_csv("parsed_hdfs_logs.csv", index=False)

print("Parsed HDFS logs saved as parsed_hdfs_logs.csv")
print(df.head())
