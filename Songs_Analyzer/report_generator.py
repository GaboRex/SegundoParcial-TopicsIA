import csv
from typing import List
from datetime import datetime

class SentimentReportGenerator:
    def __init__(self):
        self.predictions = []

    def add_prediction(self, prediction):
        self.predictions.append(prediction)

    def generate_csv_report(self):
        if not self.predictions:
            return None

        file_name = f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        fieldnames = ["url", "info", "sentiment", "execution_time", "timestamp"]

        with open(file_name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for prediction in self.predictions:
                writer.writerow({
                    "url": prediction["url"],
                    "info": prediction["info"],
                    "sentiment": prediction["sentiment"]["sentiment"],
                    "execution_time": prediction["sentiment"]["execution_time"],
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

        return file_name
