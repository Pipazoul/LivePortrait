# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from src.gradio_pipeline import GradioPipeline
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
import tyro

import boto3
from botocore.client import Config
import os
import time
tyro.extras.set_accent_color("bright_cyan")

args = tyro.cli(ArgumentConfig)

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def merge_audio_with_video(input_video, output_video):
    """Extract audio from input video and merge it with the output video."""
    # Detect audio codec
    cmd = f"ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 {input_video}"
    audio_codec = os.popen(cmd).read().strip()
    print(f"Audio codec: {audio_codec}")

    audio_path = f"audio.{audio_codec}"

    cmd = f"ffmpeg -i {input_video} -vn -acodec copy {audio_path}"
    os.system(cmd)
    print("Extracted audio", flush=True)

    # Merge audio and video
    merged_video_path = "merged.mp4"
    cmd = f"ffmpeg -i {output_video} -i {audio_path} -c:v copy -c:a aac -strict experimental {merged_video_path}"
    os.system(cmd)
    print("Merged audio and video", flush=True)

    # Clean up audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return merged_video_path


def upload_video_to_s3(s3_bucket, s3_region, s3_access_key, s3_secret_key, s3_endpoint_url, s3_use_ssl, s3_path, video_path):
    try:
        print("Uploading Video to S3...")
        # use pathstyle because minio does not support virtual host style
        s3 = boto3.client(
            's3',
            region_name=s3_region,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            endpoint_url="https://" + s3_endpoint_url,
            use_ssl=s3_use_ssl,
            config=boto3.session.Config(signature_version='s3v4'),
            verify=False
        )

        import datetime
        # create a timespamp for video name ex 5345446354.mp4
        timestamp = datetime.datetime.now().timestamp()

        s3.upload_file(
            Filename=video_path,
            Bucket=s3_bucket,
            Key=f'{s3_path}/{video_path}'
        )

        print("Done Uploading Video to S3")

        # get the url of the uploaded video for 7 days
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                 'Bucket': s3_bucket,
                 'Key': f'{s3_path}/{video_path}'
             },
            ExpiresIn=604800
        )

        return url
    except(Exception):
        print("Error: Could not create video")
        raise Exception("Error: Could not create video")
    finally:
        # Clean up downloaded files
        if os.path.exists(video_path):
            os.remove(video_path)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

        # specify configs for inference
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
        crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig
        self.gradio_pipeline = GradioPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
            args=args
        )

    def predict(
        self,
        input_image_name: str = Input(description="Name of the image file to use as input", default='image'),
        input_image_path: Path = Input(description="Path to input image"),
        input_video_path: Path = Input(description="Path to input video"),
        input_video_name: str = Input(description="Name of the video file to use as input", default='video'),
        flag_relative_input: bool = Input(description="relative pose", default=True),
        flag_do_crop_input: bool = Input(description="crop input", default=True),
        flag_remap_input: bool = Input(description="remap input", default=True),
        use_s3: bool = Input(description="Upload to S3", default=False),
        s3_bucket: str = Input(description="S3 Bucket Name", default=''),
        s3_path: str = Input(description="S3 Path", default=''),
        s3_region: str = Input(description="S3 Region", default=''),
        s3_access_key: str = Input(description="S3 Access Key", default=''),
        s3_secret_key: str = Input(description="S3 Secret Key", default=''),
        s3_endpoint_url: str = Input(description="S3 Endpoint URL", default=''),
        s3_use_ssl: bool = Input(description="S3 Use SSL", default=True),
    ) -> str:

        video_path, video_path_concat = self.gradio_pipeline.execute_video(
            str(input_image_path),
            str(input_video_path),
            flag_relative_input,
            flag_do_crop_input,
            flag_remap_input
        )

        merged_video = merge_audio_with_video(
            str(input_video_path),
            video_path
        )

        if use_s3:
            # rename[input_image_name]__[input_video_name]_timestamp.mp4
            timestamp = str(int(time.time()))
            filename = f"{input_image_name}___{input_video_name}_{timestamp}.mp4"
            os.rename(merged_video, filename)
            merged_video = filename
            print(f"Uploading {merged_video} to S3")

            url = upload_video_to_s3(
                video_path=merged_video,
                s3_bucket=s3_bucket,
                s3_path=s3_path,
                s3_region=s3_region,
                s3_access_key=s3_access_key,
                s3_secret_key=s3_secret_key,
                s3_endpoint_url=s3_endpoint_url,
                s3_use_ssl=s3_use_ssl
            )

            return url
        else:
            # return video as base64 string if not uploading to S3
            with open(merged_video, "rb") as f:
                import base64
                return base64.b64encode(f.read()).decode('utf-8')
