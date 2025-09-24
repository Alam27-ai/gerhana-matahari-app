import streamlit as st
import tempfile
import cv2
from datetime import timedelta
import os
from ultralytics import YOLO

# =====================
# Load model YOLO
# =====================
@st.cache_resource
def load_model():
    model_path = "best.pt"
    model = YOLO(model_path)
    return model

model = load_model()

# =====================
# Aturan transisi → tahapan astronomi
# =====================
TRANSITIONS = {
    ("Sun", "Partial Solar Eclipse"): "Awal gerhana matahari sebagian",
    ("Partial Solar Eclipse", "Sun"): "Akhir gerhana matahari sebagian",

    ("Partial Solar Eclipse", "Total Solar Eclipse"): "Awal gerhana matahari total",
    ("Total Solar Eclipse", "Partial Solar Eclipse"): "Akhir gerhana matahari total",

    ("Partial Solar Eclipse", "Annular Solar Eclipse"): "Awal gerhana matahari cincin",
    ("Annular Solar Eclipse", "Partial Solar Eclipse"): "Akhir gerhana matahari cincin",
}

st.title("🌑 Pencatatan Waktu Tahapan Gerhana Matahari dari Video")

# =====================
# Input user
# =====================
uploaded_video = st.file_uploader("Upload potongan video", type=["mp4", "mov", "avi"])
start_time_str = st.text_input("Masukkan waktu awal video (contoh: 12:55, 1:02:05, atau 0:15)")

# =====================
# Fungsi parsing & format waktu
# =====================
def parse_time_string(time_str):
    parts = [int(p) for p in time_str.split(":")]
    if len(parts) == 1:  # detik
        return timedelta(seconds=parts[0])
    elif len(parts) == 2:  # menit:detik
        return timedelta(minutes=parts[0], seconds=parts[1])
    elif len(parts) == 3:  # jam:menit:detik
        return timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])
    else:
        raise ValueError("Format waktu tidak valid")

def format_timestamp(td: timedelta):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# =====================
# Proses deteksi
# =====================
if uploaded_video and start_time_str:
    try:
        start_time_delta = parse_time_string(start_time_str)
    except:
        st.error("Format waktu tidak valid. Gunakan format seperti 12:55, 1:02:05, atau 0:15.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    prev_class = None
    saved_images = []

    progress_bar = st.progress(0)
    st.info("Proses deteksi dimulai (1 frame per detik)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = frame_count / fps
        # Ambil 1 frame tiap detik
        if abs(frame_time - round(frame_time)) < 1/fps:
            results = model(frame, verbose=False)

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best_box = boxes[boxes.conf.argmax()]
                current_class = model.names[int(best_box.cls)]

                if current_class != prev_class and prev_class is not None:
                    transition = (prev_class, current_class)
                    if transition in TRANSITIONS:
                        stage_name = TRANSITIONS[transition]

                        seconds_passed = round(frame_count / fps)
                        detection_time = start_time_delta + timedelta(seconds=seconds_passed)
                        detection_str = format_timestamp(detection_time)

                        # Simpan frame
                        img_filename = f"{stage_name.replace(' ', '_')}_{detection_str.replace(':', '-')}.jpg"
                        img_path = os.path.join(tempfile.gettempdir(), img_filename)
                        cv2.imwrite(img_path, frame)
                        saved_images.append((stage_name, detection_str, img_path))

                prev_class = current_class

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    os.remove(temp_video_path)

    st.success("Deteksi selesai!")

    # =====================
    # Output hasil
    # =====================
    st.write("### 📸 Tahapan Gerhana yang Terdeteksi")
    for stage_name, ts, img_path in saved_images:
        st.write(f"🕒 **{ts}** - {stage_name}")
        st.image(img_path, caption=f"{stage_name} - {ts}", use_container_width=True)
        with open(img_path, "rb") as file:
            st.download_button(
                label=f"💾 Download {stage_name} ({ts})",
                data=file,
                file_name=os.path.basename(img_path),
                mime="image/jpeg"
            )

    if saved_images:
        first_stage, first_ts, _ = saved_images[0]
        last_stage, last_ts, _ = saved_images[-1]
        st.markdown("### 📌 Kesimpulan Deteksi")
        st.info(
            f"Tahapan pertama yang terdeteksi adalah **{first_stage}** pada pukul **{first_ts}**, "
            f"dan terakhir adalah **{last_stage}** pada pukul **{last_ts}**."
        )

else:
    st.warning("Mohon upload video dan masukkan waktu awal sebelum memulai deteksi.")
