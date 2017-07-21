import os


def download_data():
    import requests, zipfile, io

    submission_file_url = "https://www.kaggle.com/c/nyc-taxi-trip-duration/download/sample_submission.zip"
    train_file_url = "https://www.kaggle.com/c/nyc-taxi-trip-duration/download/train.zip"
    test_file_url = "https://www.kaggle.com/c/nyc-taxi-trip-duration/download/test.zip"

    if not os.path.exists("./data/sample_submission.csv"):
        r = requests.get(submission_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('./data/')
        z.close()

    if not os.path.exists("./data/train.csv"):
        r = requests.get(train_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('./data/')
        z.close()

    if not os.path.exists("./data/test.csv"):
        r = requests.get(test_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('./data/')
        z.close()

download_data()