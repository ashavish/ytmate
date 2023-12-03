import cv2
import json
from PIL import Image  # pillow
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from src.scripts.settings import LOCAL_DOWNLOAD_FOLDER
from src.scripts.utils import upload_file, download_file, download_folder
from src.scripts.settings import (
    BUCKET_NAME,
    LOCAL_MODEL_FOLDER,
    DISABLE_IMAGE_CAPTIONING_PRELOAD,
)
from src import LOGGER

if not os.path.exists(f"{LOCAL_MODEL_FOLDER}/image_captioning"):
    LOGGER.info("Downloading image captioning model...")
    download_folder(
        bucket_name="ytmate-transcript-bucket",
        s3_folder=f"{LOCAL_MODEL_FOLDER}/image_captioning",
        local_folder=f"{LOCAL_MODEL_FOLDER}/image_captioning",
    )

if not DISABLE_IMAGE_CAPTIONING_PRELOAD:
    try:
        processor = BlipProcessor.from_pretrained(
            f"{LOCAL_MODEL_FOLDER}/image_captioning/processor"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            f"{LOCAL_MODEL_FOLDER}/image_captioning"
        )
        LOGGER.info("Image captioning model loaded..")
    except:
        LOGGER.info("Loading image captioning model from HuggingFace Hub")
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )


def generate_captions(
    video_path, video_id, time_interval=3, download_from_aws=False, force_train=False
):
    # load model if disabled when starting up
    if DISABLE_IMAGE_CAPTIONING_PRELOAD:
        try:
            processor = BlipProcessor.from_pretrained(
                f"{LOCAL_MODEL_FOLDER}/image_captioning/processor"
            )
            model = BlipForConditionalGeneration.from_pretrained(
                f"{LOCAL_MODEL_FOLDER}/image_captioning"
            )
            LOGGER.info("Image captioning model loaded..")
        except:
            LOGGER.info("Loading image captioning model from HuggingFace Hub")
            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
    # check if previous captions were generated
    if (not os.path.exists(f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}/images.json")) and (
        force_train == False
    ):
        download_file(
            BUCKET_NAME,
            f"{video_id}/images.json",
            f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}/images.json",
        )
        LOGGER.info("Checking if captions exist in s3")

    if (
        os.path.exists(f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}/images.json")
        and force_train == False
    ):
        LOGGER.info("Captions already exist")
        return 1

    if download_from_aws != False:
        LOGGER.info("downloading Video from s3")
        download_file(
            BUCKET_NAME,
            f"{video_id}/video.mp4",
            f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}/video.mp4",
        )
        video_path = f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}/video.mp4"

    if os.path.exists(video_path):
        LOGGER.info("Video found. Starting processing")
    else:
        LOGGER.info("Video not found. Starting processing")
        return 1

    video = cv2.VideoCapture(video_path)
    # Initialize variables
    frame_count = 0
    # output_folder = '/home/asha/Downloads/output_frames/'

    frame_count = 0
    moving_time = 3
    captured_time = []
    captions = []
    frames_per_sec = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == True:
            # Display the resulting frame
            # cv2.imshow('Frame',frame)
            # cv2.waitKey()
            frame_count = frame_count + 1
            time_in_secs = frame_count * 1.0 / frames_per_sec
            if time_in_secs > moving_time:
                LOGGER.info(f"{time_in_secs}")
                moving_time = moving_time + time_interval
                captured_time.append(time_in_secs)
                # cv2.imwrite(output_folder + f'frame_{time_in_secs}.jpg', frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_image = Image.fromarray(frame_rgb)
                text = "a scene showing"
                inputs = processor(raw_image, text=text, return_tensors="pt")
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                # LOGGER.info(f"{caption}")
                captions.append(caption)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        # Break the loop
        else:
            break

    video.release()
    LOGGER.info(f"Processed {len(captions)} frames")
    # cv2.destroyAllWindows()
    if len(captions) > 0:
        image_json = {}
        for i, j in zip(captured_time, captions):
            image_json[int(i)] = j

        if not os.path.exists(f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}"):
            os.mkdir(f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}")

        with open(f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}/images.json", "w") as file:
            json.dump(image_json, file)

        upload_file(
            BUCKET_NAME,
            f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}/images.json",
            f"{video_id}/images.json",
        )
        LOGGER.info(f"Video {video_id} processed")
    return 1
