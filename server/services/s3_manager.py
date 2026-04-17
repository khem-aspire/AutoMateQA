"""
S3 file manager for .aqa/.json test file storage with versioning.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Manager:
    """Handles upload, download, and versioning of test files in S3."""

    def __init__(self):
        self._bucket = os.getenv("AQA_S3_BUCKET", "automateqa-tests")
        endpoint_url = os.getenv("AQA_S3_ENDPOINT_URL", "").strip() or None
        self._client = boto3.client(
            "s3",
            region_name=os.getenv("AQA_S3_REGION", "ap-south-1"),
            aws_access_key_id=os.getenv("AQA_S3_ACCESS_KEY", ""),
            aws_secret_access_key=os.getenv("AQA_S3_SECRET_KEY", ""),
            endpoint_url=endpoint_url,
        )

    def _build_s3_key(self, test_id: str, version: int, filename: str) -> str:
        return f"tests/{test_id}/v{version}/{filename}"

    def upload_file(self, file_bytes: bytes, test_id: str, version: int, filename: str) -> str:
        """Upload file bytes to S3 and return the S3 key."""
        s3_key = self._build_s3_key(test_id, version, filename)
        content_type = "application/gzip" if filename.endswith(".aqa") else "application/json"
        self._client.put_object(
            Bucket=self._bucket,
            Key=s3_key,
            Body=file_bytes,
            ContentType=content_type,
        )
        logger.info("Uploaded to S3: s3://%s/%s", self._bucket, s3_key)
        return s3_key

    def download_to_tempfile(self, s3_key: str) -> str:
        """Download a file from S3 to a temp file and return the temp path."""
        suffix = ".aqa" if s3_key.endswith(".aqa") else ".json"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self._client.download_file(self._bucket, s3_key, tmp_path)
        logger.info("Downloaded S3 %s → %s", s3_key, tmp_path)
        return tmp_path

    def download_bytes(self, s3_key: str) -> bytes:
        """Download file bytes from S3."""
        response = self._client.get_object(Bucket=self._bucket, Key=s3_key)
        return response["Body"].read()

    def delete_prefix(self, prefix: str) -> int:
        """Delete all objects under an S3 prefix. Returns count deleted."""
        paginator = self._client.get_paginator("list_objects_v2")
        deleted = 0
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            objects = page.get("Contents", [])
            if not objects:
                continue
            delete_keys = [{"Key": obj["Key"]} for obj in objects]
            self._client.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": delete_keys},
            )
            deleted += len(delete_keys)
        return deleted

    def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=s3_key)
            return True
        except ClientError:
            return False


def parse_test_file(file_bytes: bytes, filename: str) -> dict:
    """Parse an .aqa (gzip) or .json file and return the TestModel dict."""
    if filename.endswith(".aqa"):
        raw = gzip.decompress(file_bytes)
        return json.loads(raw)
    else:
        return json.loads(file_bytes)
