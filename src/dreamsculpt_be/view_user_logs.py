import streamlit as st
import boto3
import json
import base64
from dotenv import load_dotenv
from pathlib import Path
import datetime as dt

# -------------------------
# ENV
# -------------------------
env_path = Path(__file__).parent / "env" / ".env.local"
load_dotenv(dotenv_path=env_path)

BUCKET = "dreamsculpt-logs"
s3 = boto3.client("s3")

# -------------------------
# UI
# -------------------------
st.set_page_config(layout="wide")
st.title("Dreamsculpt Log Timeline")

# -------------------------
# GET AVAILABLE DATES (from S3 prefixes)
# -------------------------
@st.cache_data(ttl=60)
def get_available_dates():
    paginator = s3.get_paginator("list_objects_v2")

    prefixes = set()

    for page in paginator.paginate(Bucket=BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parts = key.split("/")

            # expecting: YYYY/MM/DD/...
            if len(parts) >= 3:
                try:
                    y, m, d = parts[0], parts[1], parts[2]
                    dt.date(int(y), int(m), int(d))  # validate
                    prefixes.add(f"{y}-{m}-{d}")
                except:
                    continue

    return sorted(list(prefixes), reverse=True)


dates = get_available_dates()

if not dates:
    st.warning("No logs found")
    st.stop()

# -------------------------
# DATE PICKER (ONLY REAL DATES)
# -------------------------
selected_date_str = st.sidebar.selectbox("Select date", dates)

year, month, day = selected_date_str.split("-")
prefix = f"{year}/{month}/{day}/"

# -------------------------
# LOAD LOGS FOR DATE
# -------------------------
@st.cache_data(ttl=30)
def load_logs_for_date(prefix: str):
    paginator = s3.get_paginator("list_objects_v2")

    logs = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue

            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            data = json.loads(body)

            logs.append((key, data))

    # newest first (S3 key order + insertion order)
    return sorted(logs, key=lambda x: x[0], reverse=True)


logs = load_logs_for_date(prefix)

# -------------------------
# SCROLLABLE FEED UI
# -------------------------
st.caption(f"{len(logs)} logs for {selected_date_str}")

for key, data in logs:
    with st.container(border=True):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Text Prompt")
            st.info(data.get("text_prompt", ""))

            st.markdown("### Image Prompt")
            img_prompt = data.get("image_prompt", "")
            if img_prompt:
                img_clean = img_prompt.replace("data:image/png;base64,", "")
                try:
                    st.image(base64.b64decode(img_clean))
                except:
                    st.code(img_prompt)

            st.caption(key)

        with col2:
            st.markdown("### Generated Image")

            gen = data.get("generated_image")
            if gen:
                try:
                    st.image(base64.b64decode(gen))
                except Exception as e:
                    st.error(f"Decode error: {e}")
            else:
                st.warning("No image")