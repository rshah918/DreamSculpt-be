import json
from datetime import datetime, UTC
from dreamsculpt_be.config import USER_LOGS_S3_BUCKET

def publish_logs(s3_client, session_id: str, text_prompt: str, image_prompt: str, generated_image_b64: str):
    try:
        now = datetime.now(UTC)
        key = f"{now.year}/{now.month:02d}/{now.day:02d}/{session_id}.json"

        s3_client.put_object(
        Bucket=USER_LOGS_S3_BUCKET,
        Key=key,
        Body=json.dumps({
            'session_id': session_id,
            "text_prompt": text_prompt,
            "image_prompt" : image_prompt,
            "generated_image": generated_image_b64
        }),
        ContentType="application/json"
        )
    except Exception as e:
        print(f"s3 publish failed: {e}")