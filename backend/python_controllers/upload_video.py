import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from botocore.exceptions import ClientError

from config.s3 import s3

router = APIRouter()

# FIX: Whitelist of allowed video MIME types — reject anything else
ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-msvideo",   # .avi
    "video/x-matroska",  # .mkv
}

# FIX: Max file size limit (500 MB) to prevent abuse
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    # FIX: Validate file presence and MIME type before touching S3
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Video file is required")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: mp4, webm, mov, avi, mkv"
        )

    # FIX: Sanitize filename to avoid path traversal or special character issues
    safe_filename = os.path.basename(file.filename).replace(" ", "_")

    bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
    region = os.getenv("AWS_REGION")

    if not bucket_name or not region:
        raise HTTPException(status_code=500, detail="Server misconfiguration: S3 settings missing")

    # FIX: Check file size before uploading by reading the content
    # to prevent excessively large uploads.
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB"
        )

    # Seek back to beginning after reading for size check
    import io
    file_obj = io.BytesIO(contents)

    video_key = f"videos/{uuid.uuid4()}-{safe_filename}"

    try:
        s3.upload_fileobj(
            file_obj,
            bucket_name,
            video_key,
            ExtraArgs={
                "ContentType": file.content_type
            }
        )
    except ClientError as e:
        print(f"S3 upload error: {e}")
        raise HTTPException(status_code=500, detail="Video upload failed")

    video_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{video_key}"

    return {
        "message": "Video uploaded successfully",
        "videoKey": video_key,
        "videoUrl": video_url
    }