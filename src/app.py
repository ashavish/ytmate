from typing import Union, List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, status
from src.llm_gen import get_summary, train, get_answer, get_intent
from contextlib import asynccontextmanager
from uuid import uuid4
from src import LOGGER
from src.utils import upload_all, download_all
import glob
import os
from src.image_analysis import generate_captions
from src.sentiment_classification import classify_sentiment
from fastapi.middleware.cors import CORSMiddleware

LOGGER.info("Downloading folders from buckets")
download_all()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Text(BaseModel):
    text_json: Dict
    unique_video_id: str
    source: str  # comments /video
    force_train: bool = False


class Question(BaseModel):
    question: str
    source: str  # comments / video
    unique_video_id: str


class Video(BaseModel):
    video_path: str
    unique_video_id: str
    download_from_aws: bool = False


@app.get("/HealthCheck/", status_code=status.HTTP_200_OK)
def health_check():
    return "Health check is OK"


@app.get("/IntentCheck/", status_code=status.HTTP_200_OK)
def intent_check(question: str):
    intent_response = get_intent(question)
    return intent_response


@app.post("/CreateSummary/", status_code=status.HTTP_201_CREATED)
def get_text_summary(transcript: Text):
    text_json = transcript.text_json
    source = transcript.source
    video_id = transcript.unique_video_id
    force_train = transcript.force_train
    LOGGER.info(
        f"Request received for creating summary video_id : {video_id} , source : {source}"
    )
    summary = get_summary(text_json, source, video_id, force_train)
    LOGGER.info(f"Summary {summary}")
    return {"summary": summary, "source": source, "video_id": video_id}


@app.post("/ClassifyComments/", status_code=status.HTTP_201_CREATED)
def classify_comments(comments: Text):
    text_json = comments.text_json
    source = comments.source
    video_id = comments.unique_video_id
    force_train = comments.force_train
    LOGGER.info(
        f"Request received for comment classification for video_id : {video_id}"
    )
    emotion_percent = classify_sentiment(text_json, source, video_id, force_train)
    LOGGER.info(f"Sentiment classification {emotion_percent}")
    return emotion_percent


@app.post("/TrainQA/", status_code=status.HTTP_201_CREATED)
def train_qa(transcript: Text):
    text_json = transcript.text_json
    source = transcript.source
    video_id = transcript.unique_video_id
    force_train = transcript.force_train
    LOGGER.info(
        f"Request received for training QA for video_id : {video_id} , source : {source}"
    )
    # summary = get_summary(text_json,source,video_id,force_train)
    train_status = train(text_json, source, video_id, force_train)
    LOGGER.info("Training successful")
    return status.HTTP_200_OK


@app.post("/GetAnswer/", status_code=status.HTTP_200_OK)
def get_qa_answer(question: Question):
    question_text = question.question
    source = question.source
    video_id = question.unique_video_id
    LOGGER.info(
        f"Request received for getting answer video_id : {video_id} , source : {source}, question_text : {question_text}"
    )
    answer = get_answer(question_text, source, video_id)
    LOGGER.info(answer)
    return answer


@app.post("/GenerateImageCaptions/", status_code=status.HTTP_200_OK)
def generate_video_captions(video: Video):
    video_id = video.unique_video_id
    video_path = video.video_path
    download_from_aws = video.download_from_aws
    LOGGER.info(
        f"Request received for generating captions for video_id : {video_id}, path {video_path}"
    )
    generate_status = generate_captions(
        video_path=video_path, video_id=video_id, download_from_aws=download_from_aws
    )
    LOGGER.info("Generated captions")
    return status.HTTP_200_OK


@app.post("/UploadToBucket/", status_code=status.HTTP_200_OK)
def upload():
    LOGGER.info(f"Request received for uploading to bucket")
    upload_all()
    LOGGER.info("Files uploaded")
    return status.HTTP_200_OK


@app.post("/DownloadFromBucket/", status_code=status.HTTP_200_OK)
def download():
    LOGGER.info(f"Request received for downloading from bucket")
    download_all()
    LOGGER.info("Files downloaded")
    return status.HTTP_200_OK


@app.post("/RetrainAllLocal/", status_code=status.HTTP_200_OK)
def retrain_all():
    LOGGER.info(f"Request received for retraining all indexes")
    for video_id in glob.glob("./storage/*"):
        video_id = video_id.split("/")[-1]
        for source in glob.glob(f"./storage/{video_id}/*"):
            if os.path.isdir(source):
                source = source.split("/")[-1]
                LOGGER.info(f"Training video id {video_id}, source {source}")
                try:
                    train_status = train(
                        text_json={}, source=source, video_id=video_id, force_train=True
                    )
                except Exception as e:
                    LOGGER.error(
                        f"Error in training video id {video_id}, source {source}"
                    )
                    LOGGER.error(e)
                    continue
    LOGGER.info("All retrained")
    return status.HTTP_200_OK


if __name__ == "__main__":
    port = 3000
    app.run(debug=False, port=port, host="0.0.0.0")
