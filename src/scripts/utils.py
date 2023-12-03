import boto3
import os
from src.scripts.settings import BUCKET_NAME, LOCAL_DOWNLOAD_FOLDER
from src import LOGGER

session = boto3.Session()
s3_client = session.client("s3")


def upload_file(bucket_name, local_path, s3_path):
    s3_client.upload_file(local_path, bucket_name, s3_path)


def download_file(bucket_name, s3_path, local_path):
    s3_client.download_file(bucket_name, s3_path, local_path)


def upload_folder(bucket_name, local_folder, s3_folder):
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.join(s3_folder, os.path.relpath(local_path, local_folder))
            # s3_client.upload_file(local_path, bucket_name, s3_path)
            upload_file(bucket_name, local_path, s3_path)


def download_folder(bucket_name, s3_folder, local_folder):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
    for obj in response.get("Contents", []):
        s3_path = obj["Key"]
        local_path = os.path.join(local_folder, os.path.relpath(s3_path, s3_folder))
        if not os.path.exists(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path))
        # s3_client.download_file(bucket_name, s3_path, local_path)
        download_file(bucket_name, s3_path, local_path)


def upload_all():
    upload_folder(BUCKET_NAME, "storage", "storage")
    LOGGER.info("Uploaded all files")


def download_all():
    download_folder(BUCKET_NAME, "storage", "storage")
    LOGGER.info("Downloaded all files")


def download_source_file_from_s3(video_id, source):
    if source == "video":
        file_name = "transcript.json"
    elif source == "comments":
        file_name = "comments.json"
    elif source == "images":
        file_name = "images.json"
    LOGGER.info("Downloading json from bucket")
    if not os.path.exists(f"./{LOCAL_DOWNLOAD_FOLDER}/{video_id}"):
        os.makedirs(f"./{LOCAL_DOWNLOAD_FOLDER}/{video_id}")
    if not os.path.exists(f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}/{file_name}"):
        download_file(
            BUCKET_NAME,
            f"{video_id}/{file_name}",
            f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}/{file_name}",
        )
    return f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}/{file_name}"
