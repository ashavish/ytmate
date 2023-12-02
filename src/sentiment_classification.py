from transformers import pipeline
from collections import Counter
from src import LOGGER
from src.utils import download_source_file_from_s3, upload_file, download_folder
import os
import json
from src.settings import BUCKET_NAME, LOCAL_MODEL_FOLDER

# emotion_detection_model = pipeline(model="bhadresh-savani/distilbert-base-uncased-emotion")
# Load from local
try:
    if not os.path.exists(f"{LOCAL_MODEL_FOLDER}/sentiment_classification"):
        LOGGER.info("Downloading sentiment classification model...")
        download_folder(
            bucket_name="ytmate-transcript-bucket",
            s3_folder=f"{LOCAL_MODEL_FOLDER}/sentiment_classification",
            local_folder=f"{LOCAL_MODEL_FOLDER}/sentiment_classification",
        )

    emotion_detection_model = pipeline(
        "sentiment-analysis", model=f"{LOCAL_MODEL_FOLDER}/sentiment_classification"
    )
    LOGGER.info("Sentiment classification model loaded..")
except:
    LOGGER.info("Loading sentiment classification model from HuggingFace Hub")
    emotion_detection_model = pipeline(
        model="bhadresh-savani/distilbert-base-uncased-emotion"
    )


def classify_sentiment(text_json, source, video_id, force_train):
    source = "comments"
    # emotion_detection_model = pipeline(model="bhadresh-savani/distilbert-base-uncased-emotion")
    if os.path.exists(f"./storage/{video_id}/{source}_sentiment.json") and (
        force_train == False
    ):
        f = open(f"./storage/{video_id}/{source}_sentiment.json", "r")
        sentiment_data = json.load(f)
        emotion_percent = sentiment_data["emotion_percent"]
        LOGGER.info("Getting sentiment_data from cache")
        return emotion_percent

    if source == "comments":
        fpath = download_source_file_from_s3(video_id, source)
        data = json.load(open(fpath, "r"))
        texts = [each["commentText"] for each in data]

    LOGGER.info("Sentiment model classification started..")
    results = emotion_detection_model(texts)
    labels = [each["label"] for each in results]

    text_label = {}
    for text, label in zip(texts, labels):
        text_label[label] = text_label.get(label, []) + [text]

    cnt = len(labels)
    label_counts = Counter(labels)
    emotions = ["joy", "love", "sadness", "fear", "surprise", "anger"]
    emotion_percent = {}
    for emotion in emotions:
        emotion_percent[emotion] = label_counts.get(emotion, 0) * 1.0 / cnt * 100
    json_data = {
        "text_labels": text_label,
        "emotion_percent": emotion_percent,
    }
    if not os.path.exists(f"./storage/{video_id}"):
        os.mkdir(f"./storage/{video_id}")

    json.dump(json_data, open(f"./storage/{video_id}/{source}_sentiment.json", "w"))
    upload_file(
        BUCKET_NAME,
        f"storage/{video_id}/{source}_sentiment.json",
        f"storage/{video_id}/{source}_sentiment.json",
    )

    return emotion_percent
