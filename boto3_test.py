import boto3
import os
from botocore.config import Config

from dotenv import load_dotenv
load_dotenv()
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT"],
    aws_access_key_id=os.environ["S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    region_name="auto",
    config=Config(signature_version="s3v4"),
)

s3.put_object(
    Bucket=os.environ["S3_BUCKET"],
    Key="healthcheck.txt",
    Body=b"hello"
)

print("✅ R2 working with boto3")
