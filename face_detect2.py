import cv2
import mediapipe as mp
import numpy as np
import json
import time
from config import CAMERA_SOURCE

# -----------------------------
# Image Enhancement (from face_detect.py)
# -----------------------------
def automatic_brightness_contrast(image, clip_hist_percent=1):
    # Optimize: Use a smaller version of the image for histogram calculation
    small = cv2.resize(image, (160, 120))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0) / 2.0
    minimum_gray = 0
    while minimum_gray < hist_size and accumulator[minimum_gray] < clip_hist_percent: 
        minimum_gray += 1
        
    maximum_gray = hist_size - 1
    while maximum_gray >= 0 and accumulator[maximum_gray] >= (maximum - clip_hist_percent): 
        maximum_gray -= 1
    
    if maximum_gray <= minimum_gray or maximum_gray < 0 or minimum_gray >= hist_size:
        return image 
    
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Lowered clipLimit for speed/subtlety
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

def gamma_correction(image, gamma=1.2): # Slightly faster gamma
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_frame(frame, fast_mode=True):
    """Combines all enhancements for robust scanning."""
    if fast_mode:
        # In fast mode (for streaming), only do gamma and brightness
        enhanced = gamma_correction(frame)
        return enhanced
    
    # Full enhancement for scanning
    enhanced = automatic_brightness_contrast(frame)
    enhanced = apply_clahe(enhanced)
    return gamma_correction(enhanced)


# -----------------------------
# Brightness Score
# -----------------------------
def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


# -----------------------------
# Shine Detection
# -----------------------------
def shine_detection(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    shine_pixels = np.sum(v > 220)
    return shine_pixels / v.size


# -----------------------------
# Texture Variance
# -----------------------------
def texture_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


# -----------------------------
# Mediapipe Setup
# -----------------------------
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def interpret_skin(regions):
    """Deep analysis based on multi-region scanning."""
    desc = []
    
    # 1. Oiliness / Shine
    t_zone_shine = (regions['forehead']['shine'] + regions['nose']['shine']) / 2
    u_zone_shine = (regions['left_cheek']['shine'] + regions['right_cheek']['shine']) / 2
    
    if t_zone_shine > 0.15 and u_zone_shine > 0.15:
        desc.append("OILY: Global sebum detected.")
    elif t_zone_shine > 0.15 and u_zone_shine < 0.05:
        desc.append("COMBINATION: Oily T-Zone, dry cheeks.")
    elif t_zone_shine < 0.05 and u_zone_shine < 0.05:
        desc.append("DRY: Very low surface oil.")
    else:
        desc.append("BALANCED: Normal sebum levels.")

    # 2. Texture / Pores
    avg_texture = sum(r['texture'] for r in [regions['left_cheek'], regions['right_cheek'], regions['forehead']]) / 3
    if avg_texture > 250:
        desc.append("TEXTURED: Significant pores/breakouts.")
    elif avg_texture > 120:
        desc.append("MODERATE TEXTURE: Minor unevenness.")
    else:
        desc.append("SMOOTH: Even skin surface.")

    # 3. Redness / Sensitivity
    avg_redness = (regions['left_cheek']['redness'] + regions['right_cheek']['redness']) / 2
    if avg_redness > 10:
        desc.append("SENSITIVE: High redness/inflammation on cheeks.")
    elif avg_redness > 4:
        desc.append("MILD REDNESS: Localized flushing.")

    # 4. Skin Tone Classification
    avg_l = sum(r['lab_mean'][0] for r in [regions['left_cheek'], regions['right_cheek'], regions['forehead']]) / 3
    if avg_l > 185: tone = "vibrant fair"
    elif avg_l > 165: tone = "fair/light"
    elif avg_l > 140: tone = "medium"
    elif avg_l > 110: tone = "tan"
    else: tone = "deep"
    desc.append(f"TONE: {tone.upper()}")

    # 5. Dark Circles
    cheek_l = (regions['left_cheek']['lab_mean'][0] + regions['right_cheek']['lab_mean'][0]) / 2
    undereye_l = (regions['under_eye_left']['lab_mean'][0] + regions['under_eye_right']['lab_mean'][0]) / 2
    if undereye_l < cheek_l - 12:
        desc.append("DARK CIRCLES: Shadow detected under eyes.")

    return " | ".join(desc)

def get_region_features(frame, face_landmarks, indices):
    """Extracts features from a specific list of mediapipe landmark indices."""
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    points = []
    for idx in indices:
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    
    # Check if we have any pixels in the ROI
    if not np.any(mask > 0):
        return {"brightness": 0, "shine": 0, "texture": 0, "lab_mean": [0, 0, 0], "redness": 0}

    roi = cv2.bitwise_and(frame, frame, mask=mask)
    
    # 1. Brightness & Texture
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask_bool = mask > 0
    brightness = np.mean(gray_roi[mask_bool])
    
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
    texture = lap.var()
    
    # 2. Shine (Oiliness)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv_roi[:, :, 2]
    shine = np.sum((v > 220) & mask_bool) / np.sum(mask_bool)
    
    # 3. Skin Tone & Redness (Lab Space)
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_roi)
    
    l_mean = np.mean(l_channel[mask_bool])
    a_mean = np.mean(a_channel[mask_bool])
    b_mean = np.mean(b_channel[mask_bool])
    
    # Redness index often correlates with high 'a' channel in Lab
    # Standard 'a' for skin is around 128-150. Above that indicates redness.
    redness = max(0, a_mean - 135) 
    
    return {
        "brightness": float(brightness), 
        "shine": float(shine), 
        "texture": float(texture),
        "lab_mean": [float(l_mean), float(a_mean), float(b_mean)],
        "redness": float(redness)
    }

def extract_face_signature(face_landmarks):
    """Creates a unique list of ratios based on landmark distances (Expanded for accuracy)."""
    def dist(p1_idx, p2_idx):
        p1 = face_landmarks.landmark[p1_idx]
        p2 = face_landmarks.landmark[p2_idx]
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

    # Key landmark indices
    # 33, 263 - Outer eye corners
    # 133, 362 - Inner eye corners
    # 1 - Nose tip, 10 - Forehead top, 152 - Chin
    # 61, 291 - Mouth corners
    
    face_width = dist(33, 263)
    face_height = dist(10, 152)

    signatures = [
        dist(33, 133) / face_width,   # Left eye width ratio
        dist(362, 263) / face_width,  # Right eye width ratio
        dist(61, 291) / face_width,   # Mouth width ratio
        dist(1, 152) / face_height,   # Nose-to-chin ratio
        dist(33, 61) / face_height,   # Eye-to-mouth ratio
        dist(133, 362) / face_width,  # Inter-ocular distance
        dist(10, 1) / face_height,    # Forehead-to-nose ratio
        dist(168, 6) / face_height    # Brow-to-bridge ratio
    ]
    return [float(s) for s in signatures]

def run_face_scanner():
    face_mesh_module = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
    print("\n[CAMERA] Position your face and press 'S' to Scan or 'ESC' to Cancel.")
    
    captured_data = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only enhance every 2nd frame for display if needed, or keep it for scan
        enhanced = enhance_frame(frame)
        
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = face_mesh_module.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Analysis is done on demand ('S' key) or in VideoStream below
                interpretation = "Ready to Scan"
                
                # Draw Face Mesh
                mp_drawing.draw_landmarks(
                    image=enhanced,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Visual Feedback
                cv2.putText(enhanced, "AI ANALYSIS READY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    # Full analysis ONLY when 'S' is pressed for efficiency
                    regions = {
                        "forehead": get_region_features(enhanced, face_landmarks, [10, 67, 103, 109, 338, 297, 10]),
                        "left_cheek": get_region_features(enhanced, face_landmarks, [234, 93, 132, 58, 172, 136, 150, 234]),
                        "right_cheek": get_region_features(enhanced, face_landmarks, [454, 323, 361, 288, 397, 365, 379, 454]),
                        "nose": get_region_features(enhanced, face_landmarks, [168, 6, 197, 195, 5, 4, 168]),
                        "chin": get_region_features(enhanced, face_landmarks, [152, 148, 176, 149, 150, 136, 152]),
                        "under_eye_left": get_region_features(enhanced, face_landmarks, [101, 118, 117, 111, 116, 216]),
                        "under_eye_right": get_region_features(enhanced, face_landmarks, [330, 347, 346, 340, 345, 436])
                    }
                    interpretation = interpret_skin(regions)
                    signature = extract_face_signature(face_landmarks)
                    captured_data = {
                        "regions": regions,
                        "interpretation": interpretation,
                        "signature": signature
                    }
                    with open("skin_features.json", "w") as f:
                        json.dump(captured_data, f, indent=4)
                    print(f"✅ Full Face Scan Complete: {interpretation}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return captured_data

        cv2.imshow("AI For Her - Face Scanner", enhanced)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return None

class VideoStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
        # Set resolution for higher FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.face_mesh_module = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.latest_data = None
        self.frame_count = 0

    def get_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue
            
            self.frame_count += 1
            
            # 1. Fast enhancement for display (Low CPU)
            display_frame = enhance_frame(frame, fast_mode=True)
            
            # 2. Process face mesh (Use 640x480 for better reliability)
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh_module.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Heavy skin analysis every 15 frames
                    # Also run immediately if we don't have data yet
                    if self.frame_count % 15 == 0 or self.latest_data is None:
                        # Full enhancement for precise skin metrics
                        analysis_frame = enhance_frame(frame, fast_mode=False)
                        
                        regions = {
                            "forehead": get_region_features(analysis_frame, face_landmarks, [10, 67, 103, 109, 338, 297, 10]),
                            "left_cheek": get_region_features(analysis_frame, face_landmarks, [234, 93, 132, 58, 172, 136, 150, 234]),
                            "right_cheek": get_region_features(analysis_frame, face_landmarks, [454, 323, 361, 288, 397, 365, 379, 454]),
                            "nose": get_region_features(analysis_frame, face_landmarks, [168, 6, 197, 195, 5, 4, 168]),
                            "chin": get_region_features(analysis_frame, face_landmarks, [152, 148, 176, 149, 150, 136, 152]),
                            "under_eye_left": get_region_features(analysis_frame, face_landmarks, [101, 118, 117, 111, 116, 216]),
                            "under_eye_right": get_region_features(analysis_frame, face_landmarks, [330, 347, 346, 340, 345, 436])
                        }
                        interpretation = interpret_skin(regions)
                        self.latest_data = {
                            "regions": regions, 
                            "interpretation": interpretation, 
                            "signature": extract_face_signature(face_landmarks)
                        }

                    # Drawing landmarks
                    mp_drawing.draw_landmarks(display_frame, face_landmarks, mp_face.FACEMESH_TESSELATION, None, mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    cv2.putText(display_frame, "HIGH-FPS BIOMETRICS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if self.latest_data:
                        y_off = 55
                        for part in self.latest_data['interpretation'].split(" | "):
                            cv2.putText(display_frame, part, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            y_off += 20
            else:
                self.latest_data = None
                cv2.putText(display_frame, "SEARCHING FOR FACE...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # JPEG encoding
            ret, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    run_face_scanner()