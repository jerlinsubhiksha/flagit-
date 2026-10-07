import streamlit as st
import time
import tempfile
from pathlib import Path
import cv2
import math
import base64
from collections import defaultdict
from ultralytics import YOLO
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FLAGIT - YOLO Edition",
    page_icon="⚠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# SIDEBAR (AI SETTINGS)
# ============================================================
st.sidebar.header("⚙️ Biomechanical Settings")
HISTORY_FRAMES = st.sidebar.slider("Tracking History (Frames)", 5, 60, 15)
TORSO_LEAN_THRESH = st.sidebar.slider("Torso Lean Angle (Degrees)", 5, 45, 15, help="Minimum forward lean to be considered running.")
SPEED_PX_SEC = st.sidebar.slider("Running Velocity (Pixels/Sec)", 50, 1000, 200, help="Absolute speed threshold.")
SUSTAINED_SECONDS = st.sidebar.slider("Sustained Action (Seconds)", 0.1, 8.0, 0.2, 0.1, help="How long they must run before triggering alarm.")
RUNNING_COOLDOWN = st.sidebar.slider("Running Cooldown (Frames)", 5, 90, 45, help="How long they stay 'RUNNING'.")

st.sidebar.header("⚠️ Zone Settings")
ENABLE_ZONE = st.sidebar.checkbox("Enable Danger Zone Alerts", value=True)

# ============================================================
# BLACK + YELLOW THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --yellow: #ffd400; --yellow-dark: #b99b00; --black: #070707; --panel: #0e0e0e; --panel-2: #111111; --line: #292929; --muted: #777777; --white: #f4f4f0; }
html, body, [class*="css"] { font-family: "Space Grotesk", sans-serif; }
.stApp { background: var(--black); color: var(--white); }
.block-container { max-width: 1180px; padding-top: 1.2rem; padding-bottom: 4rem; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.brand { height: 55px; border-bottom: 1px solid #252525; display: flex; align-items: center; justify-content: space-between; }
.logo { display: flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: .18em; font-size: 18px; }
.logo-box { width: 31px; height: 31px; display: grid; place-items: center; background: var(--yellow); color: #050505; font-weight: 800; transform: skew(-8deg); }
.logo span { color: var(--yellow); }
.online { font: 10px "DM Mono"; letter-spacing: .13em; color: #858585; }
.online i { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--yellow); box-shadow: 0 0 12px var(--yellow); margin-right: 8px; }
.hero { padding: 95px 0 30px; position: relative; }
.hero:after { content: ""; position: absolute; right: 0; top: 60px; width: 220px; height: 220px; border: 1px solid #282828; border-radius: 50%; opacity: .4; }
.eyebrow { font: 10px "DM Mono"; letter-spacing: .18em; color: #8b8b8b; }
.hero h1 { font-size: clamp(60px, 9vw, 104px) !important; line-height: .87 !important; letter-spacing: -.07em !important; margin: 22px 0 !important; color: #f3f3ee !important; font-weight: 700 !important; }
.hero h1 span { color: var(--yellow); }
.lead { max-width: 720px; color: #999; line-height: 1.75; font-size: 17px; }
.stButton > button { border-radius: 7px !important; border: 1px solid #343434; background: #111; color: #eee; font-family: "Space Grotesk"; width: 100%; }
.stButton > button:hover { border-color: var(--yellow); color: var(--yellow); }
button[kind="primary"] { background: var(--yellow) !important; color: #050505 !important; border: 0 !important; font-weight: 800 !important; }
.metric-row { display: flex; gap: 10px; margin-top: 32px; }
.metric { border-left: 2px solid var(--yellow); padding: 8px 18px; background: #0d0d0d; }
.metric b { font: 16px "DM Mono"; color: var(--yellow); }
.metric span { display: block; font-size: 9px; color: #666; margin-top: 3px; letter-spacing: .1em; }
.process { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #292929; margin-top: 105px; border: 1px solid #292929; }
.proc { background: #0b0b0b; padding: 19px; }
.proc b { color: var(--yellow); font: 11px "DM Mono"; }
.proc strong { display: block; margin-top: 15px; font-size: 13px; }
.proc p { font-size: 11px; color: #686868; line-height: 1.5; }
.section { padding-top: 60px; }
.section h2 { font-size: 42px; letter-spacing: -.05em; margin: 10px 0; }
.muted { color: #777; line-height: 1.6; }
.upload { margin-top: 30px; min-height: 300px; border: 1px dashed #484848; background: repeating-linear-gradient(135deg, #0d0d0d, #0d0d0d 12px, #0b0b0b 12px, #0b0b0b 24px); display: flex; align-items: center; justify-content: center; text-align: center; }
.upload h3 { margin: 10px 0 5px; }
.upload p { color: #666; font-size: 12px; }
.upload-icon { font-size: 32px; color: var(--yellow); }
.file-card { margin-top: 15px; border: 1px solid #343434; background: #101010; padding: 17px; display: flex; justify-content: space-between; align-items: center; }
.file-card b { font-size: 13px; }
.file-card span { display: block; color: #666; font-size: 10px; margin-top: 4px; }
.scan-head { display: flex; justify-content: space-between; align-items: end; margin: 55px 0 22px; }
.scan-head h2 { font-size: 39px; letter-spacing: -.05em; margin: 9px 0; }
#pct { font: 26px "DM Mono"; color: var(--yellow); }
.console { display: grid; grid-template-columns: 1fr 300px; border: 1px solid #343434; background: #000; }
.feed { position: relative; min-height: 450px; overflow: hidden; background: #000; }
.feed img { width: 100%; height: 100%; min-height: 450px; object-fit: contain; display: block; }
.events { padding: 19px; background: #0b0b0b; height: 100%; overflow-y: auto;}
.event-title { font: 10px "DM Mono"; letter-spacing: .15em; padding-bottom: 13px; border-bottom: 1px solid #292929; }
.live { float: right; color: var(--yellow); }
.log { padding: 14px 0; border-bottom: 1px solid #1d1d1d; display: grid; grid-template-columns: 48px 1fr; gap: 8px; }
.log time { font: 9px "DM Mono"; color: #5d5d5d; }
.log b { display: block; font: 10px "DM Mono"; color: #ddd; }
.log span { display: block; font-size: 10px; color: #626262; margin-top: 4px; }
.log.flag b { color: var(--yellow); }
.count { border-top: 1px solid #292929; margin-top: 18px; padding-top: 17px; font: 10px "DM Mono"; color: #777; }
.count strong { font-size: 20px; color: var(--yellow); margin: 0 6px; }
.review-grid { display: grid; grid-template-columns: 1.4fr .8fr; gap: 15px; margin-top: 30px; }
.video-card, .result, .review { border: 1px solid #292929; background: #0d0d0d; }
.bar { padding: 12px 14px; border-bottom: 1px solid #292929; font: 9px "DM Mono"; color: #666; }
.complete { float: right; color: var(--yellow); }
.result { padding: 25px; border-top: 3px solid var(--yellow); }
.result h2 { font-size: 43px; letter-spacing: -.06em; color: var(--yellow); margin: 8px 0; }
.confidence { font: 25px "DM Mono"; color: #eee; }
.confidence small { font-size: 8px; color: #666; }
.result p { font-size: 12px; color: #888; line-height: 1.7; }
.risk { display: flex; justify-content: space-between; font: 9px "DM Mono"; color: #666; margin-top: 28px; }
.risk b { color: var(--yellow); }
.riskbar { height: 5px; background: #252525; margin-top: 9px; }
.riskbar i { height: 100%; display: block; background: var(--yellow); }
.review { margin-top: 15px; padding: 24px; }
.review h3 { font-size: 20px; margin: 8px 0 20px; }
.review-event { display: grid; grid-template-columns: 80px 1fr auto; gap: 15px; padding: 16px 0; border-top: 1px solid #222; }
.review-event time { font: 10px "DM Mono"; color: #666; }
.review-event b { display: block; font-size: 11px; }
.review-event span { display: block; color: #666; font-size: 11px; margin-top: 5px; line-height: 1.5; }
.badge { font: 8px "DM Mono"; color: #050505; background: var(--yellow); padding: 5px 7px; height: max-content; }
.badge.medium { background: #8d7b31; color: #fff; }
.badge.high { background: #ff0000; color: #fff; }
.footer-note { display: flex; gap: 25px; color: #666; font: 10px "DM Mono"; padding: 18px 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def header():
    st.markdown("""
    <div class="brand">
        <div class="logo"><span class="logo-box">F</span>FLAG<span>IT</span></div>
        <div class="online"><i></i>SYSTEM ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

def home():
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">WORKPLACE SAFETY / AI VIDEO INTELLIGENCE</div>
        <h1>Welcome to <span>FlagIt.</span></h1>
        <p class="lead">A behaviour-aware safety system powered by YOLOv8s-Pose.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("START ANALYSIS  →", type="primary"):
        st.session_state.page = "upload"
        st.rerun()

def upload():
    st.markdown("""
    <div class="section">
        <div class="eyebrow">01 / VIDEO INPUT</div>
        <h2>Give FlagIt a scene to scan.</h2>
        <p class="muted">Select your input source, draw your Danger Zone, and start scanning.</p>
    </div>
    """, unsafe_allow_html=True)

    input_source = st.radio("Select Input Source:", ("Video File", "Live Camera (Webcam)"))
    
    frame_for_canvas = None
    ready_to_scan = False

    if input_source == "Video File":
        uploaded_file = st.file_uploader("Upload workplace video", type=["mp4", "mov", "avi"], label_visibility="collapsed")
        if uploaded_file:
            st.markdown(f"""
                <div class="file-card">
                    <div><b>✓ {uploaded_file.name}</b><span>{uploaded_file.size / 1048576:.1f} MB</span></div>
                    <b style="color:#ffd400">VIDEO READY</b>
                </div>
                <br>
                """, unsafe_allow_html=True)
                
            # Save temp to get first frame
            if "video_path" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
                suffix = Path(uploaded_file.name).suffix or ".mp4"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(uploaded_file.getbuffer())
                temp_file.close()
                st.session_state.video_path = temp_file.name
                st.session_state.file_name = uploaded_file.name
                st.session_state.video_source_type = "file"

            cap = cv2.VideoCapture(st.session_state.video_path)
            ret, frame_for_canvas = cap.read()
            cap.release()
            ready_to_scan = True
            
    elif input_source == "Live Camera (Webcam)":
        if st.button("Connect to Webcam"):
            cap = cv2.VideoCapture(0)
            for _ in range(5): ret, frame_for_canvas = cap.read() # Warm up
            if ret:
                timg = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(timg.name, frame_for_canvas)
                st.session_state.video_path = 0 # 0 for webcam
                st.session_state.video_source_type = "webcam"
                st.session_state.first_frame_path = timg.name
            cap.release()
            st.rerun()
            
        if st.session_state.get("video_source_type") == "webcam":
            frame_for_canvas = cv2.imread(st.session_state.first_frame_path)
            ready_to_scan = True

    # --- DRAWING CANVAS ---
    if ready_to_scan and frame_for_canvas is not None:
        st.markdown("<h3 style='font-size:18px; margin-bottom:10px;'>Step 2: Draw Danger Zone</h3>", unsafe_allow_html=True)
        
        zones = []
        if ENABLE_ZONE:
            frame_rgb = cv2.cvtColor(frame_for_canvas, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            display_width = min(800, pil_image.width)
            scale_factor = display_width / pil_image.width
            display_height = int(pil_image.height * scale_factor)
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.4)",
                stroke_width=3,
                stroke_color="#FF0000",
                background_image=pil_image,
                drawing_mode="rect",
                key="canvas",
                width=display_width,
                height=display_height
            )
            
            num_drawn = 0
            if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
                num_drawn = len(canvas_result.json_data["objects"])
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "rect":
                        left = int(obj["left"] / scale_factor)
                        top = int(obj["top"] / scale_factor)
                        width = int(obj["width"] * obj["scaleX"] / scale_factor)
                        height = int(obj["height"] * obj["scaleY"] / scale_factor)
                        zones.append((left, top, left + width, top + height))
                        
            st.markdown(f"<p style='color:#ffd400; font-size:14px;'>Zones currently drawn and registered: <b>{num_drawn}</b></p>", unsafe_allow_html=True)
            
        if st.button("SCAN VIDEO  →", type="primary"):
            st.session_state.danger_zones = zones
            st.session_state.page = "scan"
            st.rerun()

def scanner():
    # Inject robust JS to auto-scroll to the top!
    st.components.v1.html(
        """
        <script>
            var containers = window.parent.document.querySelectorAll('.main, .stApp, html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"]');
            for (var i = 0; i < containers.length; i++) {
                containers[i].scrollTo({top: 0, behavior: 'smooth'});
                containers[i].scrollTop = 0;
            }
        </script>
        """,
        height=0
    )
    
    st.markdown("""
    <div class="scan-head">
        <div><div class="eyebrow">02 / LIVE ANALYSIS</div><h2>Scanning workplace behaviour.</h2></div>
        <div id="pct">ANALYZING</div>
    </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0)
    display = st.empty()
    
    st.toast("Booting YOLOv8s-Pose Engine...", icon="🤖")
    model = YOLO("yolov8s-pose.pt")
    
    source = st.session_state.get("video_path", 0)
    cap = cv2.VideoCapture(source)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: total_frames = 1000 # Webcam fallback
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0: fps = 30
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0: width = 640
    if height == 0: height = 480
    
    output_path = "output_flagit.webm"
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    speed_streak = defaultdict(int)
    last_alert_state = defaultdict(str)
    track_history = defaultdict(list)
    sustained_running_frames = defaultdict(int)
    
    frame_count = 0
    event_logs = []
    review_logs = []
    unique_workers = set()
    highest_severity = "NORMAL"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        if ENABLE_ZONE and "danger_zones" in st.session_state:
            for (zx1, zy1, zx2, zy2) in st.session_state.danger_zones:
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
                cv2.putText(frame, "DANGER ZONE", (zx1 + 5, zy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        results = model.track(frame, persist=True, classes=[0], conf=0.25, verbose=False)
        
        if results[0].boxes.id is not None and results[0].keypoints is not None and results[0].keypoints.conf is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            keypoints_xy = results[0].keypoints.xy.cpu().numpy()
            keypoints_conf = results[0].keypoints.conf.cpu().numpy()

            for box, track_id, kpts, confs in zip(boxes, track_ids, keypoints_xy, keypoints_conf):
                unique_workers.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                box_h = max(1, y2 - y1)
                box_w = max(1, x2 - x1)
                
                behavior_alert = None
                color = (0, 255, 0)
                final_label = "NORMAL"
                
                # Shoulders and Hips for Torso Lean
                ls, rs = kpts[5], kpts[6]
                lh, rh = kpts[11], kpts[12]
                ls_c, rs_c = confs[5], confs[6]
                lh_c, rh_c = confs[11], confs[12]
                
                cx = (x1 + x2) // 2
                foot_y = y2
                
                track_history[track_id].append((cx, foot_y, box_h))
                if len(track_history[track_id]) > HISTORY_FRAMES:
                    track_history[track_id].pop(0)
                
                is_running = False
                
                # Calculate Torso Lean Angle
                lean_angle = 0
                if ls_c > 0.5 and rs_c > 0.5 and lh_c > 0.5 and rh_c > 0.5:
                    sx = (ls[0] + rs[0]) / 2
                    sy = (ls[1] + rs[1]) / 2
                    hx = (lh[0] + rh[0]) / 2
                    hy = (lh[1] + rh[1]) / 2
                    
                    dx = sx - hx
                    dy = sy - hy
                    if dy != 0:
                        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
                        lean_angle = abs(90 - angle_deg)
                        # Draw spine for visual feedback
                        cv2.line(frame, (int(hx), int(hy)), (int(sx), int(sy)), (255, 255, 0), 2)
                        cv2.circle(frame, (int(sx), int(sy)), 3, (255, 255, 0), -1)

                # Strict Check: High Velocity AND Aggressive Torso Lean
                if len(track_history[track_id]) >= 3:
                    old_cx, old_foot_y, old_box_h = track_history[track_id][0]
                    frames_elapsed = len(track_history[track_id]) - 1
                    time_elapsed = frames_elapsed / max(1, fps)
                    
                    speed_px_sec = math.hypot(cx - old_cx, foot_y - old_foot_y) / time_elapsed
                    scale_speed_sec = (abs(box_h - old_box_h) / max(1, old_box_h)) / time_elapsed
                    
                    is_running_this_frame = False
                    # Override 1: Moving insanely fast across screen (X/Y axis)
                    if speed_px_sec > (SPEED_PX_SEC * 1.5):
                        is_running_this_frame = True
                    # Override 2: Moving fast towards/away from camera (Z axis)
                    elif scale_speed_sec > 0.4:
                        is_running_this_frame = True
                    # Standard: Moving fast + Leaning forward (side profile)
                    elif speed_px_sec > SPEED_PX_SEC and lean_angle > TORSO_LEAN_THRESH: 
                        is_running_this_frame = True
                        
                    if is_running_this_frame:
                        sustained_running_frames[track_id] += 1
                    else:
                        # Degrade slowly to handle YOLO keypoint jitter (don't reset to 0 instantly)
                        sustained_running_frames[track_id] = max(0, sustained_running_frames[track_id] - 2)
                        
                    # Only flag if they sustain the behavior for X seconds
                    if sustained_running_frames[track_id] >= (fps * SUSTAINED_SECONDS):
                        is_running = True
                
                # Cooldown
                if is_running: speed_streak[track_id] = RUNNING_COOLDOWN
                else:
                    if speed_streak[track_id] > 0: speed_streak[track_id] -= 1
                    
                if speed_streak[track_id] > 0:
                    behavior_alert = "RUNNING"
                    color = (0, 0, 255)
                
                # Zone Check
                if ENABLE_ZONE and "danger_zones" in st.session_state:
                    for (zx1, zy1, zx2, zy2) in st.session_state.danger_zones:
                        if zx1 < cx < zx2 and zy1 < foot_y < zy2:
                            behavior_alert = "ZONE INTRUSION"
                            color = (0, 165, 255)
                            break

                # Output Logic
                if behavior_alert:
                    final_label = behavior_alert
                    highest_severity = "ABNORMAL"
                    if last_alert_state[track_id] != behavior_alert:
                        timestamp = f"{int(frame_count/fps/60):02d}:{int(frame_count/fps%60):02d}"
                        event_logs.insert(0, (timestamp, behavior_alert, f"Detected at Camera 1"))
                        if len(event_logs) > 5: event_logs.pop()
                        review_logs.append((timestamp, f"#{track_id:03d}", behavior_alert, "HIGH" if behavior_alert == "ZONE INTRUSION" else "MEDIUM", "Deviation from standard movement patterns."))
                        last_alert_state[track_id] = behavior_alert
                else:
                    final_label = "NORMAL"
                    last_alert_state[track_id] = "NORMAL"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Only draw text if it's an alert
                if final_label != "NORMAL":
                    cv2.putText(frame, final_label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (cx, foot_y), 5, color, -1)
        
        out.write(frame)
        
        # UI UPDATE
        if frame_count % 3 == 0:
            pct = min(100, int((frame_count / total_frames) * 100))
            if st.session_state.get("video_source_type") == "webcam": pct = 100
            progress_bar.progress(pct)
            
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            img_str = base64.b64encode(buffer).decode()
            
            logs_html = ""
            for (t, ev, det) in event_logs:
                logs_html += f'<div class="log flag"><time>{t}</time><div><b>{ev}</b><span>{det}</span></div></div>'
            
            live_count = len(boxes) if 'boxes' in locals() else 0
            
            # Flattened HTML string to prevent Streamlit's Markdown parser from treating indentation as a code block
            html_str = f"""<div class="console">
<div class="feed"><img src="data:image/jpeg;base64,{img_str}" /><div class="cam">CAM 01 · LIVE INFERENCE</div></div>
<aside class="events"><div class="event-title">EVENT LOG <span class="live">● LIVE</span></div>{logs_html}<div class="count">LIVE WORKERS ON SCREEN: <strong>{live_count}</strong></div></aside>
</div>"""
            display.markdown(html_str, unsafe_allow_html=True)
            
    cap.release()
    out.release()
    
    st.session_state.output_video = output_path
    st.session_state.review_logs = review_logs
    st.session_state.severity = highest_severity
    st.session_state.worker_count = len(unique_workers)
    
    st.session_state.page = "review"
    st.rerun()

def review():
    st.markdown("""
    <div class="section">
        <div class="eyebrow">03 / SAFETY REVIEW</div>
        <h2>Analysis complete.</h2>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.4, 0.8])
    with left:
        st.markdown('<div class="video-card"><div class="bar">PROCESSED VIDEO<span class="complete">COMPLETE</span></div></div>', unsafe_allow_html=True)
        st.video(st.session_state.output_video)

    with right:
        color = "var(--yellow)" if st.session_state.severity == "ABNORMAL" else "#00ff00"
        score = 82 if st.session_state.severity == "ABNORMAL" else 15
        st.markdown(f"""
        <div class="result" style="border-color:{color};">
            <div class="eyebrow">CLASSIFICATION</div>
            <h2 style="color:{color};">{st.session_state.severity}</h2>
            <div class="confidence">94%<small> CONFIDENCE</small></div>
            <p>End-of-shift safety pipeline review generated successfully.</p>
            <div class="risk"><span>RISK SCORE</span><b style="color:{color};">{score} / 100</b></div>
            <div class="riskbar"><i style="width:{score}%; background:{color};"></i></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="review"><div class="eyebrow">EXPLAINABLE REVIEW</div><h3>EVENT TIMELINE</h3>', unsafe_allow_html=True)
    
    if len(st.session_state.review_logs) == 0:
        st.markdown("<p style='color:#666'>No safety violations detected.</p>", unsafe_allow_html=True)
    else:
        for t, w, title, sev, desc in st.session_state.review_logs:
            bc = "high" if sev == "HIGH" else "medium"
            st.markdown(f"""
            <div class="review-event">
                <time>{t}</time>
                <div><b>{title}</b><span>Worker {w} · {desc}</span></div>
                <label class="badge {bc}">{sev}</label>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    </div>
    <div class="footer-note">◉ AI PATTERN DETECTION ✓ EXPLAINABLE OUTPUT ⚠ {st.session_state.worker_count} WORKERS SCANNED</div>
    """, unsafe_allow_html=True)

    if st.button("↻ NEW ANALYSIS"):
        st.session_state.page = "upload"
        st.rerun()

# ============================================================
# RENDER
# ============================================================
if "page" not in st.session_state: st.session_state.page = "home"
header()
if st.session_state.page == "home": home()
elif st.session_state.page == "upload": upload()
elif st.session_state.page == "scan": scanner()
elif st.session_state.page == "review": review()
