# Fingertip-Only OCR — EasyOCR-only + SIM-history warp
# (OP/INFO 모드 스케줄 재구성 & HUD 개선)
# --------------------------------------------------------------

import cv2, time, numpy as np, threading, queue, os, math, re, traceback
from PIL import Image, ImageDraw, ImageFont
from tts_reader import TTSReader
import seeing

try:
    from rapidfuzz import process, fuzz
    from jamo import h2j, j2hcj
    _SPELLFIX_OK = True
except Exception as _e:
    print(f"[SPELLFIX] disabled: { _e }")
    _SPELLFIX_OK = False

# ========= User / Display =========
CAMERA_ID = 2
CAPTURE_TARGET_W = 1920
CAPTURE_TARGET_H = 1080
WORK_WIDTH_TARGET = 1280
DISPLAY_MAX_W = 1280
WINDOW_NAME = 'Assistive Fingertip OCR (fast)'

# ========= OCR / Scheduling =========
OCR_ENABLED = True
BASE_OCR_PERIOD  = 1.5
EXTRA_OCR_PERIOD = 0.6
STALE_AGE_SEC    = 7.0
LOW_CONF_TH      = 0.55

# ========= ROI (work-space) =========
# <실험1> 문제 상황: 서연 세탁기 글씨가 작아서 detection 못함
# [CASE 1] ROI_W, ROI_H = 420, 420 -> detection 성능 향상 (부족함)
# [CASE 2] 서연 세탁기 사진 잘라서 글씨 더 크게 보이도록 조정 (안 해봄)
# [CASE 3] MAX_OCR_LONG 을 420으로 제한하지 않고 원본을 넣기 (속도 느려짐)
# [CASE 4] 실제 사이즈로 인쇄

ROI_W, ROI_H = 420,420
MIN_ROI_W, MIN_ROI_H = 200, 120
BLUR_VAR_THRESH = 80.0

# ROI 유지 유예(손가락 잠깐 끊겨도 ROI 내부 TTL 갱신)
ROI_KEEPALIVE_GRACE_SEC = 1.2
last_roi_active_until = 0.0
# ========= No masking =========
#USE_MASKED_FULL_ROI = True 삭제(8.20)
# EXCLUDE_PAD = 8
# MASK_FILL_VAL = (127,127,127)

# ========= Donut OCR (unused) =========
# DONUT_PAD = 3  삭제(8.20)
# SUBROI_MIN_AREA = 1200
# MAX_SUBROIS = 1

# ========= TTL / Pruning =========
BASE_TTL = 3.0 # 연장시간 조정(8.20)
PIN_GRACE_SEC = 1.2
MAX_OVERLAYS = 300
#ONSCREEN_KEEPALIVE = 0.8 삭제(8.20)
HARD_MAX_LIFETIME = 9.0
no_repeat_until_ts = 2.0 # 같은 문장 재발화 금지 시간 (8.21)

IGNORE_HARD_CAP_WHILE_FINGER_IN_ROI = True
PRUNE_TIMEOUT_SEC =0.5 # prune 주기 변수화 (8.20)

# ========= Merge criteria =========
MERGE_IOU_TH = 0.50
MERGE_CENTER_DIST = 28.0

# ========= TTS =========
TTS_ENABLE = True
TTS_CONF = 0.0 # 발화 기준 임계치 필요할듯. 지금은 다 말함 (8.20)
TTS_REPEAT_SEC = 1.0
# TTS_QUEUE_MAX = 1 삭제(8.20)
TTS_TARGET_STICKY_SEC = 0.6
# TTS_DEBUG = False 삭제 (8.20)
# TTS_STRICT_LATEST = True 삭제(8.20)

STRICT_DICT_ONLY = True
TTS_CONF_FALLBACK = 0.35

SHOW_TTS_HINT = True
tts_current_display = ""
tts_current_note = ""
tts_last_spoken_text = ""   # <<< CHANGED: 마지막 발화 문구를 HUD에 유지

# ★ 추가: 모드 전환 멘트 직후 1회 즉시 요약 트리거 + 선점 락
INFO_FORCE_IMMEDIATE = False
tts_force_lock = threading.Lock()

# ========= Speed knobs =========
# 수정1: MAX_OCR_LONG 416 -> 420
MAX_OCR_LONG = 420
ENHANCE_MODE = "off"
MOTION_GATE_PX = 2.0
MAX_TEXT_DRAW = 30

# ========= Global tracking (SIM) =========
FLOW_DS = 0.45
FLOW_MAX_CORNERS=240; FLOW_QUALITY=0.01; FLOW_MIN_DISTANCE=7
FLOW_WINSIZE=(21,21); FLOW_LEVELS=3
RESEED_INTERVAL_FRAMES=8

MAX_TRANS_PX = 90
MAX_SCALE_STEP = 0.18
MAX_ROT_STEP_DEG = 10.0
EMA_ALPHA_SIM = 0.28

USE_ORB_FALLBACK = True
ORB_NFEATURES=600; ORB_MIN_GOOD=45

# ========= Finger =========
EMA_ALPHA_FINGER=0.35
FINGER_STALE_MS = 800
finger_last_seen = 0.0
# had_finger = False 삭제(8.20)
last_finger_xy = None

# ========= YOLO =========
YOLO_DEBUG = True
YOLO_DRAW_ALL = True
YOLO_IMG_SIZE = 640
YOLO_CONF_TH  = 0.25
YOLO_IOU_TH   = 0.50
YOLO_CLASS_NAME = None
YOLO_CLASS_ID = 0
# fingerip_o.pt 사용 가능
YOLO_WEIGHTS  = r'weights/fingertip.pt'

YOLO_SHOW_INPUT = False
YOLO_INPUT_WIN  = 'YOLO_INPUT'

DO_PIC=True # 보기모드 입력 출력(8.21)
# # ========= speed change parameters =========
# # 해상도/스케일
# WORK_WIDTH_TARGET = 960
# YOLO_IMG_SIZE     = 448
# MAX_OCR_LONG      = 360
# FLOW_DS           = 0.35

# # ROI 크기
# ROI_W, ROI_H = 270,270

# # 빈도/주기
# BASE_OCR_PERIOD   = 2.0
# EXTRA_OCR_PERIOD  = 0.9
# RESEED_INTERVAL_FRAMES = 12
# PRUNE_TIMEOUT_SEC = 1.0

# # 전역 SIM
# FLOW_MAX_CORNERS  = 150
# FLOW_WINSIZE      = (17,17)
# FLOW_LEVELS       = 2
# # estimateAffinePartial2D maxIters ~800로 하향

# # KLT
# KLT_LEVELS        = 2
# KLT_WIN           = (25,25)
# KLT_TERM          = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 12, 0.03)
# KLT_N_SAMPLES     = 6
# KLT_USE_CLAHE     = False

# # YOLO
# YOLO_CONF_TH      = 0.3  # 잡음↓
# # yolo_model.predict(..., half=True)  # (GPU일 때)

# # OCR
# # rotation_info=[0] 로 축소
# # canvas_size=1280, mag_ratio=1.1
# MAX_TEXT_DRAW     = 30
# MAX_OVERLAYS      = 150


# ========= KLT fallback =========
# 수정2: False -> True
USE_KLT_FALLBACK = True
KLT_WIN=(31,31); KLT_LEVELS=4
KLT_TERM=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 20, 0.03)
KLT_FB_MAX=4.0; KLT_ERR_MAX=100.0; KLT_STEP_MAX=30.0
KLT_OUT_MARGIN=4; KLT_N_SAMPLES=12; KLT_RING_R=10
# KLT_RESEED_EVERY=6; 삭제(8.20) 
KLT_MIN_GOOD=5; KLT_LOSS_GRACE=3
KLT_USE_CLAHE=True

klt_pts_prev=None; klt_lost_frames=0; #frames_since_reseed=0 삭제 (8.20)

# ========= OCR time meter =========
OCR_EMA=None; OCR_EMA_ALPHA=0.25

# ==== Dict-based merge parameters ====
# 수정3: 80 -> 60
DICT_MERGE_SCORE = 70 # 사전 변환 수정(8.21)
DICT_TIE_DELTA   = 3
DICT_ONLY =True # 사전 단어만 표기(8.21) 

# ========= GUIDE MODE =========
GUIDE_MODE = False
GUIDE_TARGET = None
GUIDE_TOL_PX = 40
GUIDE_REPEAT_SEC = 1.0
GUIDE_LAST_TS = 0.0
GUIDE_LAST_SENT = ""
GUIDE_TARGET_ITEM = None
GUIDE_REQUIRE_FINGER = True

# ========= INFO/OP 모드 =========
MODE_OP   = 1   # 조작 모드
MODE_INFO = 2   # 보기 모드
MODE_GUIDE = 3  # 안내 모드

mode_lock = threading.Lock()
mode_state = MODE_OP

# 보기 주기(초)
INFO_PERIOD_SEC = 5.0    # <<< CHANGED: 8s → 5s

# 즉시 실행/주기 스케줄용
_next_info_due = 0.0

# 보기용 최신 프레임 공유
_latest_frame_for_info = None
_latest_frame_lock = threading.Lock()

# 보기 스레드 제어
_info_stop = threading.Event()

def _is_speaker_busy() -> bool:
    try:
        import pygame
        return pygame.mixer.music.get_busy()
    except Exception:
        return False

def _say_once(text: str):
    """한 문장만 안전하게 재생(비동기 TTS) + 표시 유지.
    - 재생 시작을 잠깐 대기(최대 2s 시도)
    - 끝날 때까지 폴링(최대 30s), 그 후 target만 None으로 지워 재반복 차단
    - HUD는 tts_last_spoken_text로 마지막 발화를 계속 보여줌
    """
    global no_repeat_until_ts

    t_start = time.time()                     # ★ 누락되었던 t_start 보완
    set_tts_target(text)
    no_repeat_until_ts = time.time() + 60.0   # 같은 문장 재enqueue 금지(안전 마진)

    # 재생 시작 감지(최대 2s)
    while not _info_stop.is_set():
        if _is_speaker_busy():
            break
        if '_last_spoken_enqueue_ts' in globals() and _last_spoken_enqueue_ts >= t_start:
            time.sleep(0.1)
            break
        if (time.time() - t_start) > 2.0:
            break
        time.sleep(0.02)

    # 재생 종료 대기(최대 30s)
    t0 = time.time()
    while _is_speaker_busy() and not _info_stop.is_set():
        if (time.time() - t0) > 30.0:
            break
        time.sleep(0.05)

    # target만 지워서 재반복 방지(표시는 tts_last_spoken_text로 유지됨)
    set_tts_target(None)
    no_repeat_until_ts = 0.0

def announce_force_async(text: str, after=None):
    """모드 전환 전용: 현재 재생 중이어도 즉시 중단하고 text부터 발화."""
    def _runner():
        with tts_force_lock:
            if TTS_ENABLE and tts is not None:
                try: tts.clear_queue()
                except Exception: pass
                for m in ("stop","cancel","flush"):
                    if hasattr(tts, m):
                        try: getattr(tts, m)()
                        except Exception: pass
            _say_once(text)   # 마지막 멘트 HUD 유지 + 재반복 억제 로직 그대로
            if callable(after):
                try: after()
                except Exception: pass
    threading.Thread(target=_runner, daemon=True).start()

def _enter_op_mode():
    # 1) 지금 말하는 TTS 전부 끊고, 2) "조작 모드로 전환합니다"를 끝까지 말한 다음, 3) 모드 적용
    def _after():
        global mode_state
        with mode_lock:
            mode_state = MODE_OP
        globals().update(GUIDE_MODE=False)
    announce_force_async("조작 모드로 전환합니다.", after=_after)

def _enter_info_mode():
    # 1) 선점 발화 → 2) 발화 끝난 뒤 INFO 모드 플래그 세팅 + 첫 요약 즉시 허용
    def _after():
        global mode_state, _next_info_due, INFO_FORCE_IMMEDIATE
        with mode_lock:
            mode_state = MODE_INFO
            _next_info_due = 0.0        # 진입 직후 1회 즉시
        INFO_FORCE_IMMEDIATE = True     # 다음 루프에서 바로 요약
        globals().update(GUIDE_MODE=False)
    announce_force_async("보기 모드로 전환합니다. 지금부터 상황을 설명합니다.", after=_after)
    

def _enter_guide_mode():
    # 1) 선점 발화 → 2) 발화 완료 후 GUIDE 모드 적용
    def _after():
        global mode_state, GUIDE_MODE
        with mode_lock:
            mode_state = MODE_GUIDE
            GUIDE_MODE = True
    announce_force_async("안내 모드로 전환합니다. 목표를 지정해 주세요.", after=_after)


# def _finger_present_now() -> bool: #손가락 탐지 제거 (8.21)
#     try:
#         if last_finger_xy is None:
#             return False
#         return (time.time() - finger_last_seen) * 1000.0 <= FINGER_STALE_MS
#     except NameError:
#         return False

# def _wait_till_no_finger(max_wait_sec: float = 8.0):
#     t0 = time.time()
#     while _finger_present_now() and not _info_stop.is_set():
#         if time.time() - t0 > max_wait_sec:
#             break
        # time.sleep(0.05)

def _info_worker():
    """보기 모드: 진입 즉시 1회, 이후 5초마다. 말하는 중이면 '말 끝 + 2초' 후 실행
       단, 모드 진입 멘트 직후 1회는 지연 없이 곧바로 요약."""
    global _next_info_due, INFO_FORCE_IMMEDIATE

    while not _info_stop.is_set():
        time.sleep(0.05)

        with mode_lock:
            info_on = (mode_state == MODE_INFO)
        if not info_on:
            _next_info_due = 0.0
            continue

        now = time.time()
        if now < _next_info_due:
            continue

        # 1) 말하는 중이면 끝날 때까지 대기
        was_busy = False
        while _is_speaker_busy() and not _info_stop.is_set():
            was_busy = True
            time.sleep(0.05)

        # 1-1) 일반 경우: 말 끝났으면 2초 뒤로
        #      단, 직전이 "진입 멘트"였다면 지연 없이 곧바로 진행
        if was_busy:
            if INFO_FORCE_IMMEDIATE:
                # 진입 멘트 방금 끝남 → 즉시 1회 실행
                INFO_FORCE_IMMEDIATE = False
            else:
                _next_info_due = time.time() + 2.0
                continue

        # 2) 손가락 있으면 치워 달라고 말하고(한번) 손가락 사라질 때까지 대기
        
        
        # 3) 최신 프레임 요약
        with _latest_frame_lock:
            frame = None if _latest_frame_for_info is None else _latest_frame_for_info.copy()

        if frame is not None:
            try:
                summary = seeing.summarize_scene(frame, easy_reader,do_pic=DO_PIC, debug_dir=r"logs/ocr_bbox",debug_font=r"C:\Windows\Fonts\malgun.ttf")
            except Exception as e:
                print("[INFO] summarize failed:", e)
                summary = None

            if summary:
                _say_once(summary)

        # 4) 다음 실행 예약(지금 시점 + 5s)
        _next_info_due = time.time() + INFO_PERIOD_SEC

# ===== STT =====
USE_STT = True
try:
    import speech_recognition as sr
    _STT_OK = True
except Exception as _e:
    print(f"[STT] disabled: {_e}")
    _STT_OK = False

# ========= GPU / OCR / YOLO 로드 =========
def torch_cuda_ok():
    try:
        import torch
        ok = bool(torch.cuda.is_available())
        print(f"[GPU] torch CUDA available: {ok}")
        return ok
    except Exception as e:
        print(f"[GPU] torch check failed: {e}")
        return False

gpu_ok = torch_cuda_ok()

OCR_ENGINE=None; easy_reader=None
import easyocr
try:
    easy_reader = easyocr.Reader(['ko'], gpu=gpu_ok,
                                 model_storage_directory='models',
                                 user_network_directory='user_network',
                                 recog_network='best_accuracy1',
                                 download_enabled=False)
    OCR_ENGINE = 'easyocr_gpu' if gpu_ok else 'easyocr_cpu'
    print(f"[OCR] EasyOCR (GPU={gpu_ok})")
except Exception as e:
    traceback.print_exc()
    raise SystemExit("No OCR engine available")

# === 보기(상황 설명) 스레드 기동 ===
def _start_info_thread_once():
    if not hasattr(_start_info_thread_once, "_started"):
        threading.Thread(target=_info_worker, daemon=True).start()
        _start_info_thread_once._started = True
_start_info_thread_once()

# ========= YOLO =========
try:
    from ultralytics import YOLO
    yolo_device = 0 if gpu_ok else 'cpu'
    yolo_model = YOLO(YOLO_WEIGHTS)
    print(f"[YOLO] Loaded: {YOLO_WEIGHTS} (device={yolo_device})")
    class_names = yolo_model.names
    if YOLO_CLASS_NAME:
        inv = {str(v).lower(): int(k) for k, v in class_names.items()}
        if YOLO_CLASS_NAME.lower() in inv:
            YOLO_CLASS_ID = inv[YOLO_CLASS_NAME.lower()]
except Exception as e:
    traceback.print_exc()
    raise SystemExit("[YOLO] 모델 로드 실패. YOLO_WEIGHTS 경로/파일 확인")

def _pick_best_tip(cands, last_xy):
    if not cands: return None
    if last_xy is None:
        return max(cands, key=lambda t: t[2])
    lx, ly = last_xy
    def score(t):
        cx, cy, conf, _ = t
        d2 = (cx-lx)**2 + (cy-ly)**2
        return conf - 0.0005*d2
    return max(cands, key=score)

# --- YOLO 비동기 워커 ---
yolo_in_q=queue.Queue(maxsize=1); yolo_out_q=queue.Queue(maxsize=1); yolo_stop=threading.Event()
def _yolo_worker():
    while not yolo_stop.is_set():
        try:
            frame = yolo_in_q.get(timeout=0.2)
        except queue.Empty:
            continue
        yolo_in_vis, _r, _off = _yolo_letterbox_bgr(frame, YOLO_IMG_SIZE)
        res = yolo_model.predict(source=frame, imgsz=YOLO_IMG_SIZE,
                                 conf=YOLO_CONF_TH, iou=YOLO_IOU_TH,
                                 device=yolo_device, verbose=False)
        det=None; raw_boxes=[]
        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            cands=[]
            for b in res[0].boxes:
                x1,y1,x2,y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                cls_id = int(b.cls[0]) if b.cls is not None else 0
                raw_boxes.append((x1,y1,x2,y2,conf,cls_id))
                if YOLO_CLASS_ID is not None and cls_id != YOLO_CLASS_ID: continue
                cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                cands.append((cx, cy, conf, (x1, y1, x2-x1, y2-y1)))
            best=_pick_best_tip(cands, last_finger_xy)
            if best is not None:
                cx, cy, conf, (x,y,w,h) = best
                det={'xy':(int(round(cx)), int(round(cy))),
                     'box':(int(x), int(y), int(w), int(h)),
                     'conf':conf, 'ts':time.time(),
                     'raw_boxes':raw_boxes, 'yolo_in':yolo_in_vis}
        else:
            det={'xy':None, 'raw_boxes':[], 'yolo_in':yolo_in_vis}
        try:
            while True: yolo_out_q.get_nowait()
        except queue.Empty:
            pass
        try: yolo_out_q.put_nowait(det)
        except queue.Full: pass
threading.Thread(target=_yolo_worker, daemon=True).start()

def _yolo_letterbox_bgr(img, new_size=YOLO_IMG_SIZE, pad_val=114):
    h, w = img.shape[:2]
    r = min(new_size / float(h), new_size / float(w))
    new_w, new_h = int(round(w*r)), int(round(h*r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (new_size - new_w) / 2.0; dh = (new_size - new_h) / 2.0
    left, right = int(math.floor(dw)), int(math.ceil(dw))
    top, bottom = int(math.floor(dh)), int(math.ceil(dh))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             borderType=cv2.BORDER_CONSTANT,
                             value=(pad_val, pad_val, pad_val))
    return out, r, (left, top)

# ========= TTS =========
# tts_q=queue.Queue(maxsize=TTS_QUEUE_MAX) 삭제(8.20)
# tts_is_speaking=threading.Event() 삭제(8.20)
tts_stop=threading.Event()
tts_target_lock=threading.Lock()
tts_target_text=None
_last_spoken_enqueue_ts=0.0
# tts_last_done_ts=0.0 삭제(8.20)
tts_last_seen_target_ts=0.0

SPELLFIX_ENABLE=True
JAMO_THRESHOLD=85; JAMO_THRESHOLD_LOWCONF=80
# 수정4: 딕셔너리 확장 (서연 세탁기 포함하도록)
#"동작","일시정지",
DICT_WORDS=["통살균","원격제어","예약","내마음","세탁","헹굼","탈수","물온도","빨래추가","알림음","구김방지","터보샷", "강력물살","온수세탁","냉수세탁","물높이","코스","동작/일시정지","전원"]
CANON={"표준세탁":"세탁","손세탁":"세탁"}

def _to_jamo(s: str) -> str:
    try:
        return j2hcj(h2j(s))
    except Exception:
        return s or ""

if _SPELLFIX_OK:
    _DICT_JAMO=[_to_jamo(w) for w in DICT_WORDS]
else:
    _DICT_JAMO=[]

_TOKENIZER=re.compile(r"[가-힣A-Za-z0-9]+|[^\s가-힣A-Za-z0-9]")

def correct_token(tok: str, threshold: int):
    if not (_SPELLFIX_OK and SPELLFIX_ENABLE and _DICT_JAMO and tok):
        return tok, 0.0
    q=_to_jamo(tok)
    res=process.extractOne(q, _DICT_JAMO, scorer=fuzz.ratio, score_cutoff=threshold)
    if not res: return tok, 0.0
    matched, score, idx=res
    best=DICT_WORDS[idx]; best=CANON.get(best, best)
    return best, float(score)

def correct_text(text: str, threshold: int):
    if not (_SPELLFIX_OK and SPELLFIX_ENABLE and _DICT_JAMO and text):
        return text, False
    out=[]; changed=False
    for tok in _TOKENIZER.findall(text):
        if re.match(r"^[가-힣A-Za-z0-9]+$", tok):
            fixed, sc = correct_token(tok, threshold=threshold)
            if fixed!=tok: changed=True
            out.append(fixed)
        else:
            out.append(tok)
    return "".join(out), changed

DICT_SPEAK_ENABLE=True
DICT_THRESHOLD=80; DICT_THRESHOLD_LOWCONF=80
def _build_dict_index(words, canon_map):
    keys=[]; vals=[]
    for w in words: keys.append(w); vals.append(canon_map.get(w,w))
    for alias, canon in canon_map.items(): keys.append(alias); vals.append(canon)
    keys_j=[_to_jamo(re.sub(r"\s+","",k)) for k in keys]
    return keys, keys_j, vals
_DICT_KEYS, _DICT_KEYS_J, _DICT_VALS=_build_dict_index(DICT_WORDS, CANON)
# _DICT_KEYS_PLAIN=[re.sub(r"\s+","",k).casefold() for k in _DICT_KEYS] 삭제(8.20)
def _normalize_plain(s:str)->str: return re.sub(r"\s+","",(s or "")).casefold()
def map_to_dict_canon(text: str, threshold: int): #사용
    if not DICT_SPEAK_ENABLE or not text: return None, 0.0
    if _SPELLFIX_OK:
        queries=[]
        s=re.sub(r"\s+","",text)
        if s: queries.append(_to_jamo(s))
        for tok in _TOKENIZER.findall(text):
            if re.match(r"^[가-힣A-Za-z0-9]+$", tok): queries.append(_to_jamo(tok))
        best_idx, best_sc=-1, 0.0
        for q in queries:
            res=process.extractOne(q, _DICT_KEYS_J, scorer=fuzz.ratio, score_cutoff=threshold)
            if res:
                _, sc, idx=res
                if sc>best_sc:
                    best_sc=float(sc); best_idx=int(idx)
        if best_idx>=0: return _DICT_VALS[best_idx], best_sc

    # 수정6: 사전 매칭 점수로만 결정. 부분집합 때문에 옵션을 읽어버리는 문제(강 -> 강력세탁 매칭) 방지.
    # q_full=_normalize_plain(text)
    # q_tokens=[_normalize_plain(tok) for tok in _TOKENIZER.findall(text) if re.match(r"^[가-힣A-Za-z0-9]+$", tok)]
    # for q in [q_full]+q_tokens:
    #     if not q: continue
    #     for i,k in enumerate(_DICT_KEYS_PLAIN):
    #         if q==k: return _DICT_VALS[i], 100.0
    # for q in [q_full]+q_tokens:
    #     if not q: continue
    #     for i,k in enumerate(_DICT_KEYS_PLAIN):
    #         if (k and k in q) or (q and q in k): return _DICT_VALS[i], 90.0
    return None, 0.0

def enrich_with_dict(text: str, conf: float):
    canon, sc = map_to_dict_canon(text, threshold=DICT_MERGE_SCORE)
    display = canon if canon else text
    return display, canon, float(sc or 0.0), float(conf or 0.0)

def _has_korean(s: str)->bool:
    return any('가'<=ch<='힣' for ch in (s or ""))

# (tts / 안내 스레드 근처 아무 곳에 추가)
# def announce_async(text: str): 삭제(8.20)
#     threading.Thread(target=_say_once, args=(text,), daemon=True).start() 


# TTS 초기화
try:
    import tempfile, os
    try:
        tts=TTSReader(cooldown_sec=TTS_REPEAT_SEC, speaking_rate=1.05, pitch=0.0,
                      min_len=2, credentials_path=r"yugpae-4f8335e15ba0.json",
                      cache_dir=None, persist_cache=False)
    except TypeError:
        tts=TTSReader(cooldown_sec=TTS_REPEAT_SEC, speaking_rate=1.05, pitch=0.0,
                      min_len=2, credentials_path=r"yugpae-4f8335e15ba0.json")
        for attr in ("set_cache","disable_cache"):
            if hasattr(tts, attr):
                try: getattr(tts, attr)(persist=False, dir=None)
                except Exception: pass
    try:
        if not (hasattr(tts,"cache_dir") and getattr(tts,"cache_dir") is None):
            tmp_cache=os.path.join(tempfile.gettempdir(),"tts_runtime_cache")
            os.makedirs(tmp_cache, exist_ok=True)
            if hasattr(tts,"cache_dir"): tts.cache_dir=tmp_cache
    except Exception: pass
except Exception as e:
    print(f"[TTS] init failed: {e}")
    TTS_ENABLE=False
    tts=None

try:
    import pygame
    if not pygame.mixer.get_init(): pygame.mixer.init()
    pygame.mixer.music.set_volume(1.0)
except Exception: pass

def tts_scheduler():
    global _last_spoken_enqueue_ts, tts_last_spoken_text, no_repeat_until_ts #tts_last_done_ts 삭제(8.20)
    last_sent_text = None
    while not tts_stop.is_set():
        time.sleep(0.05)
        if not TTS_ENABLE or tts is None:
            continue

        with tts_target_lock:
            tgt = (tts_target_text or "").strip()

        # 타겟이 없으면 아무것도 하지 않고 넘김 (중단/정지 금지)
        if not tgt:
            last_sent_text = None
            continue

        now = time.time()

        # 지금 말하는 중이면 일반 TTS는 절대 선점/중단하지 않음
        if _is_speaker_busy():
            continue

        # 같은 문장을 너무 자주 반복하지 않음
        if tgt == last_sent_text and now < no_repeat_until_ts:
            continue

        # 재생 (모드 전환이 아닌 한 clear_queue/stop/flush 절대 금지)
        try:
            tts.say(tgt)
            tts_last_spoken_text = tgt
            _last_spoken_enqueue_ts = now
            #tts_last_done_ts = now 삭제(8.20)
            last_sent_text = tgt
        except Exception as e:
            print(f"[TTS] error: {e}")



if TTS_ENABLE:
    threading.Thread(target=tts_scheduler, daemon=True).start()

# def set_tts_target(text_or_none, note: str=""):
#     global tts_target_text, tts_current_display, tts_current_note
#     # 일반 TTS는 오직 타겟만 갱신. 여기서 재생을 중단/선점하지 않음.
#     with tts_target_lock:
#         tts_target_text = text_or_none
#         tts_current_display = (text_or_none or "").strip()
#         tts_current_note = note or ""
last_text="" #선점 발화를 위한 마지막 text 기록 (8.21)

def set_tts_target(text_or_none, note: str="", # 선점 발화를 위한 force 추가 (8.21)
                   *, force: bool=False):
    """TTS 타겟 갱신.
    - force=True        : 지금 재생 중단(큐 비우고 stop/cancel/flush) 후 새 타겟 적용
    - bypass_repeat=True: 같은 문장 반복 억제 타이머 무시(바로 재발화)
    - speak_now=True    : 스케줄러 기다리지 않고 즉시 say() 실행
    """
    global tts_target_text, tts_current_display, tts_current_note
    global no_repeat_until_ts, _last_spoken_enqueue_ts
    global last_text
    # 1) 타겟 갱신
    with tts_target_lock:
        tts_target_text = text_or_none
        tts_current_display = (text_or_none or "").strip()
        tts_current_note = note or ""

    # 2) 선점 옵션
    if force and ("tts" in globals()) and (tts is not None) and last_text != text_or_none:
        last_text=text_or_none
        try:
            if hasattr(tts, "clear_queue"): tts.clear_queue()
            for m in ("stop","cancel","flush"):
                if hasattr(tts, m):
                    try: getattr(tts, m)()
                    except Exception: pass
        except Exception:
            pass
        _last_spoken_enqueue_ts = 0.0  # 스케줄러와 동기화

        


# ========= Camera =========
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW) if cv2.getBuildInformation().find('Windows')!=-1 else cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened(): raise SystemExit("카메라 열기 실패")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_TARGET_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_TARGET_H)
cap.set(cv2.CAP_PROP_FPS, 30)
try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except: pass
time.sleep(0.15)
Wc=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hc=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[Camera] requested ~{CAPTURE_TARGET_W}x{CAPTURE_TARGET_H}, actual {Wc}x{Hc}")

WORK_SCALE=min(1.0, WORK_WIDTH_TARGET/float(Wc))
print(f"[Work] WORK_SCALE={WORK_SCALE:.3f} (work width ~{int(Wc*WORK_SCALE)})")

# ========= State =========
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
frame_idx=0
prev_gray_s=None; prev_pts=None
overlays=[]; last_prune=time.time()

# OCR 스케줄
last_ocr_time=0.0
last_roi=None

# ORB
orb=None; bf=None
if USE_ORB_FALLBACK:
    orb=cv2.ORB_create(nfeatures=ORB_NFEATURES)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ========= Utils =========
_next_oid=1
def alloc_oid():
    global _next_oid
    oid=_next_oid; _next_oid+=1; return oid

def clamp_rect(x,y,w,h,W,H):
    x=max(0,min(x,W-1)); y=max(0,min(y,H-1))
    w=max(1,min(w,W-x)); h=max(1,min(h,H-y))
    return x,y,w,h

def poly_center(poly): return np.mean(poly,axis=0)

def bbox_of_poly(poly):
    x1=float(np.min(poly[:,0])); y1=float(np.min(poly[:,1]))
    x2=float(np.max(poly[:,0])); y2=float(np.max(poly[:,1]))
    return (x1,y1,x2-x1,y2-y1)

def variance_of_laplacian(g): return cv2.Laplacian(g, cv2.CV_64F).var()

def rect_contains(outer, inner, tol=2.0):
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (ix >= ox - tol) and (iy >= oy - tol) and \
           (ix + iw <= ox + ow + tol) and (iy + ih <= oy + oh + tol)

def _canon_equal(a: str, b: str) -> bool:
    a = (a or "").strip(); b = (b or "").strip()
    if not a or not b: return False
    try:
        return _normalize_plain(a) == _normalize_plain(b)
    except Exception:
        import re
        aa = re.sub(r"\s+","",a).casefold()
        bb = re.sub(r"\s+","",b).casefold()
        return aa == bb


def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ax2,ay2=ax+aw,ay+ah; bx2,by2=bx+bw,by+bh
    ix1,iy1=max(ax,bx),max(ay,by)
    ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih; union=aw*ah+bw*bh-inter+1e-9
    return inter/union

# def expand_rect(x,y,w,h,pad,W,H): 삭제(8.20)
#     x2=x-pad; y2=y-pad; w2=w+2*pad; h2=h+2*pad
#     return clamp_rect(x2,y2,w2,h2,W,H)

def is_visible_in_view(poly, W, H, min_overlap=0.7):
    x, y, w, h = bbox_of_poly(poly)
    x1, y1, x2, y2 = x, y, x+w, y+h
    vx1, vy1, vx2, vy2 = 0, 0, W, H
    ix1, iy1 = max(x1, vx1), max(y1, vy1)
    ix2, iy2 = min(x2, vx2), min(y2, vy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih; area  = max(1.0, w * h)
    return (inter / area) >= min_overlap

def draw_overlays(frame, items, now_ts):
    H, W = frame.shape[:2]
    to_draw=[]
    for it in items:
        if is_visible_in_view(it['poly'], W, H, min_overlap=0.7):
            it['last_seen']=now_ts
            to_draw.append(it)
    to_draw=to_draw[:MAX_TEXT_DRAW]
    for it in to_draw:
        cv2.polylines(frame, [it['poly'].astype(int)], True, (255,165,0), 2, cv2.LINE_AA)

    img_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil=Image.fromarray(img_rgb); draw=ImageDraw.Draw(pil)
    font_path=None
    for p in [r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\NanumGothic.ttf",
              r"C:\Windows\Fonts\NotoSansCJKkr-Regular.otf",
              "/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.ttc"]:
        if os.path.isfile(p): font_path=p; break
    font=ImageFont.truetype(font_path, 22) if font_path else ImageFont.load_default()

    for it in to_draw:
        poly=it['poly'].astype(int)
        x=int(np.min(poly[:,0])); y=int(np.min(poly[:,1]))-6
        draw.text((x, max(0,y)), f"{it['text']} ({it['conf']:.2f})",
                  font=font, fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0))
    frame[:]=cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def prune_overlays(items, now, active_roi=None):
    def center_in_roi(c, roi):
        if roi is None: return False
        rx,ry,rw,rh = roi
        return (rx<=c[0]<=rx+rw) and (ry<=c[1]<=ry+rh)

    kept=[]
    for it in items:
        pinned = (now <= it.get('pin_until', 0.0))
        if pinned:
            kept.append(it); continue
        birth = it.get('time', now)
        alive_by_ttl = (now <= it.get('expiry', 0.0))
        if IGNORE_HARD_CAP_WHILE_FINGER_IN_ROI and active_roi is not None:
            c = poly_center(it['poly'])
            if center_in_roi(c, active_roi):
                if alive_by_ttl:
                    kept.append(it)
                continue
        under_hard_cap = ((now - birth) <= HARD_MAX_LIFETIME)
        if alive_by_ttl and under_hard_cap:
            kept.append(it)

    if len(kept) > MAX_OVERLAYS:
        kept = sorted(
            kept,
            key=lambda d: max(d.get('expiry', 0.0), d.get('pin_until', 0.0)),
            reverse=True
        )[:MAX_OVERLAYS]
    return kept

# ========= SIM helpers =========
def closest_rotation(A):
    U, _, Vt = np.linalg.svd(A); R = U @ Vt
    if np.linalg.det(R) < 0: Vt[-1,:]*=-1; R = U @ Vt
    return R
def project_to_similarity(M): #사용
    A=M[:,:2]; R=closest_rotation(A)
    s=float(np.trace(A.T@R)/2.0); t=M[:,2].reshape(2)
    return s, R, t
def angle_from_R(R): return math.atan2(R[1,0], R[0,0]) #사용
def build_similarity(s, theta):
    c, n = math.cos(theta), math.sin(theta)
    A=np.array([[c,-n],[n,c]], dtype=np.float32)*float(s)
    return A

SIM_HIST_MAX=240
sim_steps=[]

def _rect_aabb_after_M(rect, M3, W, H):
    x,y,w,h=rect
    corners=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
    tc=(corners @ M3[:2,:2].T)+M3[:2,2]
    minx,miny=float(np.min(tc[:,0])), float(np.min(tc[:,1]))
    maxx,maxy=float(np.max(tc[:,0])), float(np.max(tc[:,1]))
    rx=int(max(0,minx)); ry=int(max(0,miny))
    rh=int(max(1,min(H-1,maxy)-ry)); rw=int(max(1,min(W-1,maxx)-rx)) #약간의 오류 수정 (8.21)
    return (rx,ry,rw,rh)

def estimate_similarity_small(prev_gray_s, gray_s, prev_pts): #사용
    if prev_pts is None or len(prev_pts) < 140:
        prev_pts=cv2.goodFeaturesToTrack(prev_gray_s, maxCorners=FLOW_MAX_CORNERS,
                                         qualityLevel=FLOW_QUALITY, minDistance=FLOW_MIN_DISTANCE, blockSize=7)
    if prev_pts is None: return None, None
    next_pts, st, err=cv2.calcOpticalFlowPyrLK(prev_gray_s, gray_s, prev_pts, None,
                                               winSize=FLOW_WINSIZE, maxLevel=FLOW_LEVELS,
                                               criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,12,0.03))
    if next_pts is None: return None, None
    P=prev_pts[st==1].reshape(-1,1,2); Q=next_pts[st==1].reshape(-1,1,2)
    if len(P) < 60: return None, None
    M,_=cv2.estimateAffinePartial2D(P,Q,method=cv2.RANSAC,
                                    ransacReprojThreshold=3.0, maxIters=1500, confidence=0.99)
    if M is None: return None, None
    return M, next_pts

def transform_overlays_similarity(items, s, theta, t_s): #사용
    tx=float(t_s[0])/FLOW_DS; ty=float(t_s[1])/FLOW_DS
    step_mag=math.hypot(tx,ty)
    if step_mag>MAX_TRANS_PX:
        scale=MAX_TRANS_PX/(step_mag+1e-6)
        tx*=scale; ty*=scale
    A=build_similarity(s, theta).astype(np.float32)
    for it in items:
        pts=it['poly'].astype(np.float32)
        it['poly']=(pts@A.T)+np.array([tx,ty], dtype=np.float32)

def orb_similarity(prev_g, cur_g):
    kp1, des1 = orb.detectAndCompute(prev_g, None)
    kp2, des2 = orb.detectAndCompute(cur_g, None)
    if des1 is None or des2 is None or len(kp1)<8 or len(kp2)<8: return None
    matches=bf.knnMatch(des1, des2, k=2)
    good=[]
    for mn in matches:
        if len(mn)==2:
            m,n=mn
            if m.distance < 0.75*n.distance: good.append(m)
    if len(good) < ORB_MIN_GOOD: return None
    src=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M,_=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,
                                    ransacReprojThreshold=3.0,maxIters=1500,confidence=0.99)
    return M

# ========= KLT =========
def _build_gray_for_klt(gray): #사용
    g=gray
    if KLT_USE_CLAHE:
        clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g=clahe.apply(g)
    gx=cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy=cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag=cv2.magnitude(gx,gy)
    if mag.max()>0: mag=(mag/mag.max())*255.0
    return mag.astype(np.uint8)

def _klt_seed_ring(center, n=KLT_N_SAMPLES, r=KLT_RING_R): #사용
    cx, cy = float(center[0]), float(center[1])
    pts=[(cx,cy)]
    for k in range(n):
        a=2.0*math.pi*k/float(n)
        pts.append((cx+r*math.cos(a), cy+r*math.sin(a)))
    return np.array(pts, dtype=np.float32).reshape(-1,1,2)

def _in_bounds(pt, W, H, margin=0):
    x,y=float(pt[0]), float(pt[1])
    return (-margin<=x<=(W-1+margin)) and (-margin<=y<=(H-1+margin))

def klt_track_multi(prev_gray, cur_gray, prev_pts, W, H): #사용
    if prev_gray is None or cur_gray is None or prev_pts is None or len(prev_pts)==0:
        return None, None
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_pts, None,
                                           winSize=KLT_WIN, maxLevel=KLT_LEVELS, criteria=KLT_TERM)
    if p1 is None: return None, None
    p0r, st2, err2 = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, p1, None,
                                              winSize=KLT_WIN, maxLevel=KLT_LEVELS, criteria=KLT_TERM)
    good=[]
    for i in range(len(prev_pts)):
        if st[i]==1 and st2[i]==1:
            fb=float(np.linalg.norm(prev_pts[i,0]-p0r[i,0]))
            e=float(err[i][0]) if err is not None else 0.0
            step=float(np.linalg.norm(p1[i,0]-prev_pts[i,0]))
            if fb<=KLT_FB_MAX and e<=KLT_ERR_MAX and step<=KLT_STEP_MAX and _in_bounds(p1[i,0], W, H, KLT_OUT_MARGIN):#KTL_OUT_MARGIN 단순화 (8.20)
                good.append(p1[i,0])
    if len(good)<KLT_MIN_GOOD: return None, None
    good=np.array(good, dtype=np.float32)
    med=np.median(good, axis=0)
    cx, cy = int(round(float(med[0]))), int(round(float(med[1])))
    if not _in_bounds((cx,cy), W, H, 0): return None, None
    return (cx,cy), good.reshape(-1,1,2)

# ===== Donut / merge utils =====
def rect_from_poly(poly):#사용
    x,y,w,h=bbox_of_poly(poly); return (int(x),int(y),int(w),int(h))

def fingertip_overlaps_box(finger, box): #사용
    if finger is None: return False
    x,y,w,h=box
    return (x<=finger[0]<=x+w) and (y<=finger[1]<=y+h)

def clip_poly_to_rect(poly, rect):
    x,y,w,h=rect; rx1,ry1,rx2,ry2=x,y,x+w,y+h
    P=poly.copy()
    P[:,0]=np.clip(P[:,0], rx1, rx2); P[:,1]=np.clip(P[:,1], ry1, ry2)
    return P

def merge_update_overlays(items, new_items, roi_rect, now_ts,
                          iou_th=MERGE_IOU_TH, center_dist_th=MERGE_CENTER_DIST):
    rx, ry, rw, rh = roi_rect

    def center_in_roi(c):
        return (rx <= c[0] <= rx+rw) and (ry <= c[1] <= ry+rh)

    roi_indices = [idx for idx, it in enumerate(items) if center_in_roi(poly_center(it['poly']))]
    used_old = set()

    for ni in new_items:
        poly_new = clip_poly_to_rect(ni['poly'], roi_rect)
        box_new  = bbox_of_poly(poly_new)
        raw_txt  = str(ni.get('text','')).strip()
        raw_conf = float(ni.get('conf', 0.0))
        disp_new, canon_new, csc_new, conf_new = enrich_with_dict(raw_txt, raw_conf)

        best_idx = -1
        best_iou = -1.0
        best_d   = 1e9

        for idx in roi_indices:
            if idx in used_old:
                continue
            it = items[idx]
            box_old = bbox_of_poly(it['poly'])

            # ① 위치기반 매칭(IoU/센터거리)
            i = iou(box_new, box_old)
            cxn = (box_new[0]*2 + box_new[2]) * 0.5
            cyn = (box_new[1]*2 + box_new[3]) * 0.5
            cxo = (box_old[0]*2 + box_old[2]) * 0.5
            cyo = (box_old[1]*2 + box_old[3]) * 0.5
            d = math.hypot(cxn - cxo, cyn - cyo)
            loc_match = (i >= iou_th) or (d <= center_dist_th)

            # ② 같은 글자 + 포함관계면 매칭으로 간주(작은 박스가 큰 박스 안에 있는 경우 등)
            text_same = _canon_equal(it.get('canon_text') or it.get('text'),
                                     canon_new or disp_new)
            contained = rect_contains(box_old, box_new) or rect_contains(box_new, box_old)
            text_same_contained = text_same and contained

            if not (loc_match or text_same_contained):
                continue

            # 베스트 선택(우선 IoU, 다음 거리)
            if (i > best_iou) or (abs(i - best_iou) < 1e-6 and d < best_d):
                best_iou, best_d, best_idx = i, d, idx

        if best_idx >= 0:
            it = items[best_idx]
            # 우선순위: (사전 일치 점수) > (conf)
            csc_old = float(it.get('canon_score', 0.0))
            conf_old = float(it.get('conf', 0.0))

            replace = False

            if csc_new >= DICT_MERGE_SCORE and csc_old < DICT_MERGE_SCORE:
                replace = True
            elif csc_new >= DICT_MERGE_SCORE and csc_old >= DICT_MERGE_SCORE:
                if csc_new > csc_old + DICT_TIE_DELTA:
                    replace = True
                elif abs(csc_new - csc_old) <= DICT_TIE_DELTA and conf_new > conf_old:
                    replace = True
            else:
                if conf_new > conf_old and csc_new > csc_old: #신뢰도가 더 높을 경우만 대체 (8.20)
                    replace = True                            #상대비교 기반으로 대체하면 어떨지 고민

            if replace:
                it['poly']        = poly_new
                it['ocr_text']    = raw_txt
                it['text']        = disp_new
                it['canon_text']  = canon_new
                it['canon_score'] = csc_new
                it['conf']        = conf_new
                it['expiry']      = now_ts + BASE_TTL
            #연장시간 코드 중첩 삭제 (8.20)

            used_old.add(best_idx)

        else:
            if DICT_ONLY and (disp_new is None or disp_new not in DICT_WORDS): #사전 단어만 표기 (8.21)
                continue
            items.append({
                'poly':        poly_new,
                'ocr_text':    raw_txt,
                'text':        disp_new,
                'canon_text':  canon_new,
                'canon_score': csc_new,
                'conf':        conf_new,
                'time':        now_ts,
                'last_seen':   now_ts,
                'expiry':      now_ts + BASE_TTL,
                'pin_until':   0.0,
                'id':          alloc_oid()
            })

    # ROI 안에 있던 기존 항목들의 여유시간(keepalive) 연장
    # 메인루프에서 연장되므로 시간 연장 삭제(8.20)
    # prune_overlays에서 사용하는것과 겹침 삭제(8.20)
    # 만약 바운딩 박스가 많아지면 여기서 prune 한번 진행 필요
    return items


def dedupe_same_text_overlays(items, iou_th=0.55, center_dist_th=26.0):
    """동일/유사 텍스트(사전 정규화 기준) 중복 박스 제거.
    - 같은 텍스트로 간주되는 박스가 서로 많이 겹치거나 가깝거나
      한쪽이 다른쪽을 '포함'하면 하나만 남김
    - 우선순위: (1) 사전 일치 점수 높음 → (2) 동률이면 conf 높은 것
    """
    def _canon_key(it):
        t = (it.get('canon_text') or it.get('text') or '').strip()
        try:
            return _normalize_plain(t)
        except Exception:
            import re as _re
            return _re.sub(r"\s+","",t).casefold()

    def _rect(it):
        return bbox_of_poly(it['poly'])

    def _score(it):
        csc = float(it.get('canon_score', 0.0))
        conf = float(it.get('conf', 0.0))
        return ((1 if csc >= DICT_MERGE_SCORE else 0), csc, conf)

    groups = {}
    for it in items:
        key = _canon_key(it)
        if not key:  # 빈 문자열 제외
            continue
        groups.setdefault(key, []).append(it)

    keep = set()
    drop = set()
    for key, arr in groups.items():
        arr_sorted = sorted(arr, key=_score, reverse=True)
        for i, a in enumerate(arr_sorted):
            if id(a) in drop or id(a) in keep:
                continue
            keep.add(id(a))
            ax, ay, aw, ah = _rect(a)
            acx, acy = ax+aw*0.5, ay+ah*0.5
            for b in arr_sorted[i+1:]:
                if id(b) in drop or id(b) in keep:
                    continue
                bx, by, bw, bh = _rect(b)
                bcx, bcy = bx+bw*0.5, by+bh*0.5
                ov = iou((ax,ay,aw,ah), (bx,by,bw,bh))
                d  = ((acx-bcx)**2 + (acy-bcy)**2)**0.5
                contained = rect_contains((ax,ay,aw,ah), (bx,by,bw,bh)) or rect_contains((bx,by,bw,bh), (ax,ay,aw,ah))
                if contained or (ov >= iou_th) or (d <= center_dist_th):
                    drop.add(id(b))

    if not drop:
        return items
    return [it for it in items if id(it) not in drop]


# ===== GUIDE MODE 유틸 =====
def _overlay_center(it):
    P = it['poly']
    x1, y1 = float(np.min(P[:,0])), float(np.min(P[:,1]))
    x2, y2 = float(np.max(P[:,0])), float(np.max(P[:,1]))
    return (0.5*(x1+x2), 0.5*(y1+y2))

def _choose_target_overlay(target_canon: str, overlays, finger_xy=None):
    cands = []
    t = (target_canon or "").strip()
    if not t: return None
    for it in overlays:
        ct = (it.get('canon_text') or "").strip()
        tx = (it.get('text') or "").strip()
        ok = (ct == t) or (tx == t) or (t in tx)
        if ok:
            cx, cy = _overlay_center(it)
            d = 0.0
            if finger_xy is not None:
                d = math.hypot(cx - (finger_xy[0]), cy - (finger_xy[1]))
            canon_bonus = 1.0 if (ct == t) else 0.0
            cands.append((canon_bonus, float(it.get('conf',0.0)), -d, it))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][3]

def _dir_sentence(dx, dy):
    def q(px):
        a = abs(int(round(px)))
        if a < 30: lvl = "조금"
        elif a < 90: lvl = "약간"
        elif a < 180: lvl = "보통"
        else: lvl = "많이"
        return lvl, a
    msg = []
    if dx > 0: lvl, a = q(dx); msg.append(f"오른쪽으로 {a}픽셀({lvl})")
    elif dx < 0: lvl, a = q(dx); msg.append(f"왼쪽으로 {a}픽셀({lvl})")
    if dy > 0: lvl, a = q(dy); msg.append(f"아래로 {a}픽셀({lvl})")
    elif dy < 0: lvl, a = q(dy); msg.append(f"위로 {a}픽셀({lvl})")
    return " , ".join(msg) if msg else "그대로 유지"

def set_guide_target_from_text(text: str):
    global GUIDE_TARGET, GUIDE_TARGET_ITEM
    if not text:
        GUIDE_TARGET = None
        GUIDE_TARGET_ITEM = None
        set_tts_target("목표가 비었습니다.", note="guide")
        return False
    canon, sc = map_to_dict_canon(text, threshold=DICT_THRESHOLD)
    if not canon:
        set_tts_target(f"'{text}'는 사전에 없습니다.", note="guide no-dict")
        return False
    GUIDE_TARGET = canon
    GUIDE_TARGET_ITEM = None
    set_tts_target(f"목표 '{canon}' 안내를 시작합니다.", note="guide")
    return True

def guide_tick(now_ts, finger_xy, overlays):
    global GUIDE_LAST_TS, GUIDE_LAST_SENT, GUIDE_TARGET_ITEM, GUIDE_TARGET

    if not GUIDE_MODE or not GUIDE_TARGET:
        return

    if GUIDE_REQUIRE_FINGER and finger_xy is None:
        if now_ts - GUIDE_LAST_TS >= GUIDE_REPEAT_SEC:
            set_tts_target("손가락을 화면에 올려 주세요.", note="guide")
            GUIDE_LAST_TS = now_ts
            GUIDE_LAST_SENT = "ask_finger"
        return

    tgt = _choose_target_overlay(GUIDE_TARGET, overlays, finger_xy)
    GUIDE_TARGET_ITEM = tgt

    if tgt is None:
        if now_ts - GUIDE_LAST_TS >= 2.0:
            set_tts_target(f"화면에서 '{GUIDE_TARGET}'을 찾지 못했습니다.", note="guide")
            GUIDE_LAST_TS = now_ts
            GUIDE_LAST_SENT = "not_found"
        return

    cx, cy = _overlay_center(tgt)
    if finger_xy is None:
        if now_ts - GUIDE_LAST_TS >= GUIDE_REPEAT_SEC:
            set_tts_target(f"목표 '{GUIDE_TARGET}'이 화면에 있습니다. 손가락을 이동해 주세요.", note="guide")
            GUIDE_LAST_TS = now_ts
            GUIDE_LAST_SENT = "where_only"
        return

    dx = int(round(cx - finger_xy[0]))
    dy = int(round(cy - finger_xy[1]))
    dist = math.hypot(dx, dy)

    if dist <= GUIDE_TOL_PX:
        if GUIDE_LAST_SENT != "arrived":
            set_tts_target(f"도착. '{GUIDE_TARGET}' 입니다.", note="guide ok")
            GUIDE_LAST_SENT = "arrived"
            GUIDE_LAST_TS = now_ts
        return

    if (now_ts - GUIDE_LAST_TS) >= GUIDE_REPEAT_SEC:
        msg = _dir_sentence(dx, dy)
        set_tts_target(f"{msg}", note=f"guide d={int(dist)}")
        GUIDE_LAST_SENT = msg
        GUIDE_LAST_TS = now_ts

def highlight_guide_target(frame_bgr, item):
    if item is None: return
    poly = item['poly'].astype(int)
    cv2.polylines(frame_bgr, [poly], True, (0,0,255), 3, cv2.LINE_AA)
    cx, cy = map(int, _overlay_center(item))
    cv2.circle(frame_bgr, (cx,cy), 6, (0,0,255), -1)

def stt_listen_once(timeout=4, phrase_time_limit=4):
    if not (USE_STT and _STT_OK):
        set_tts_target("음성 인식이 비활성화되어 있습니다.", note="stt off")
        return None
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            set_tts_target("목표 단어를 말씀해 주세요.", note="stt")
            if hasattr(r, "adjust_for_ambient_noise"):
                r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        try:
            text = r.recognize_google(audio, language="ko-KR")
        except Exception:
            text = r.recognize_google(audio, language="ko-KR")
        return text
    except Exception as e:
        print(f"[STT] error: {e}")
        set_tts_target("음성 인식에 실패했습니다.", note="stt err")
        return None

# ===== OCR worker =====
task_q=queue.Queue(maxsize=1)
result_q=queue.Queue(maxsize=2)

def enhance_for_ocr(bgr):
    if ENHANCE_MODE=="off": return bgr
    if ENHANCE_MODE=="fast":
        blur=cv2.GaussianBlur(bgr,(0,0),0.8)
        return cv2.addWeighted(bgr, 1.6, blur, -0.6, 0)
    img=bgr.copy()
    img=cv2.bilateralFilter(img, d=0, sigmaColor=45, sigmaSpace=12)
    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB); L,A,B=cv2.split(lab)
    clahe=cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8,8)); L=clahe.apply(L)
    img=cv2.cvtColor(cv2.merge([L,A,B]), cv2.COLOR_LAB2BGR)
    blur=cv2.GaussianBlur(img,(0,0),0.9)
    return cv2.addWeighted(img, 1.8, blur, -0.8, 0)

def prep_fixed(roi_bgr):
    h,w=roi_bgr.shape[:2]
    long_side=max(h,w); scale=min(1.0, float(MAX_OCR_LONG)/float(long_side))
    proc=cv2.resize(roi_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale<1.0 else roi_bgr
    proc=enhance_for_ocr(proc)
    sx_pre=proc.shape[1]/float(w); sy_pre=proc.shape[0]/float(h)
    return proc, sx_pre, sy_pre

def run_ocr_rect(frame_work, rect_work, mask_boxes=None):
    x,y,w,h=rect_work
    src=frame_work[y:y+h, x:x+w].copy()
    base_off=(x,y)
    proc,sx_pre,sy_pre=prep_fixed(src)
    items=[]
    r1=easy_reader.readtext(proc, detail=1, decoder='greedy',
                            rotation_info=[0,180],
                            contrast_ths=0.05, adjust_contrast=0.7,
                            text_threshold=0.6, low_text=0.3, link_threshold=0.4,
                            canvas_size=1920, mag_ratio=1.3,
                            paragraph=False, min_size=2)
    for (bbox_points,text,prob) in r1:
        poly=np.array(bbox_points,dtype=np.float32)
        poly[:,0]=poly[:,0]/sx_pre + base_off[0]
        poly[:,1]=poly[:,1]/sy_pre + base_off[1]
        bx,by,bw,bh=bbox_of_poly(poly)
        if bw*bh>=120: items.append({'poly':poly,'text':text,'conf':float(prob)})
    return items

def ocr_worker():
    while True:
        item=task_q.get()
        if item is None: break
        t0=time.time()
        out=[]
        for rect_work in item['rects']:
            out.extend(run_ocr_rect(item['frame_work'], rect_work, mask_boxes=None))
        dt_ms=(time.time()-t0)*1000.0
        result_q.put({
            'roi': item['roi'],
            'new_items': out,
            'dt_ms': dt_ms,
            'frame_idx': item['frame_idx'],
        })
threading.Thread(target=ocr_worker, daemon=True).start()

def drain_queue(q):
    try:
        while True: q.get_nowait()
    except queue.Empty:
        pass

# === 폴백 요약기 === 삭제 (8.20)
# def _fallback_summarize(frame_bgr):
#     try:
#         r = easy_reader.readtext(frame_bgr, detail=1)
#         tokens = [re.sub(r"[^가-힣0-9A-Za-z]", "", t).strip() for (_b,t,_c) in r]
#         tokens = [t for t in tokens if t]
#         if not tokens:
#             return "눈에 띄는 텍스트가 없습니다."
#         top = ", ".join(tokens[:5])
#         return f"화면에서 텍스트가 보입니다: {top}"
#     except Exception:
#         return "장면을 요약할 수 없습니다."

# ===== Main loop =====
print("실시간 시작. 'q' 종료 / 'o' OCR ON/OFF / 't' HUD / 's' TTS / 'y' YOLO 입력 / 'p' YOLO PNG 저장")
print("모드 전환: '1' 조작 모드 / '2' 보기 모드(상황 설명)")# / '3' 안내 모드(목표로 이동 안내)")  # <<< CHANGED
print("ROI 조절: '[' 너비-, ']' 너비+, ';' 높이-, \"'\" 높이+ / 'r' 기본값 복원")
#print("GUIDE: '3' 안내 모드 / 'v' 음성으로 목표 지정 / 'f' 문자 입력 / 'c' 목표 취소")

s_ema=1.0; theta_ema=0.0; tx_ema=0.0; ty_ema=0.0
prev_gray_full=None; prev_gray_klt=None
prev_gray_s=None; prev_pts=None
# <실험2> 손가락 없을 때 TTS 잘못 안내
# [CASE 1] KLT OFF & FINGER_STALE_MS 800 -> 2000 (손가락 잘 안 따라올 수 있음)
# [CASE 2] KLT ON & (1초 내내 KLT만 썼으면 finger_is_fresh = False)
# [CASE 3] YOLO 연속 n번 해야 KLT ON

# 수정7: KLT 단독 추적 시작 시간 기록 (위 상황의 CASE2에 해당)
klt_only_start_ts = 0.0  

finger_src="NONE"; yolo_last_conf=None; klt_draw_pts=None; yolo_box_count=None; yolo_last_in=None

while True:
    ret, frame_cap = cap.read()
    if not ret: break
    
    frame_work = frame_cap if WORK_SCALE==1.0 else cv2.resize(frame_cap, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
    H,W = frame_work.shape[:2]
    frame_for_ocr=frame_work.copy()
    frame_disp=frame_work.copy()

    # 안내용 최신 프레임 공유
    with _latest_frame_lock:
        _latest_frame_for_info = frame_work.copy()

    gray=cv2.cvtColor(frame_work, cv2.COLOR_BGR2GRAY)
    gray_klt=_build_gray_for_klt(gray) if USE_KLT_FALLBACK else gray

    # ---- Global SIM ----
    gray_s=cv2.resize(gray, None, fx=FLOW_DS, fy=FLOW_DS, interpolation=cv2.INTER_AREA)
    M_s=None; did_motion=False
    if prev_gray_s is not None:
        reseed=((frame_idx % RESEED_INTERVAL_FRAMES)==0)
        if reseed: prev_pts=None
        M_s, next_pts = estimate_similarity_small(prev_gray_s, gray_s, prev_pts)
        prev_pts=next_pts
    prev_gray_s=gray_s

    if M_s is not None:
        s_step,R_step,t_step_s=project_to_similarity(M_s)
        s_step=max(1.0-MAX_SCALE_STEP, min(1.0+MAX_SCALE_STEP, s_step))
        theta_step=angle_from_R(R_step)
        theta_step=max(-math.radians(MAX_ROT_STEP_DEG), min(math.radians(MAX_ROT_STEP_DEG), theta_step))
        s_ema=(1-EMA_ALPHA_SIM)*s_ema+EMA_ALPHA_SIM*s_step
        theta_ema=(theta_ema+((theta_step-theta_ema+math.pi)%(2*math.pi)-math.pi)*EMA_ALPHA_SIM)
        tx_ema=(1-EMA_ALPHA_SIM)*tx_ema+EMA_ALPHA_SIM*float(t_step_s[0])
        ty_ema=(1-EMA_ALPHA_SIM)*ty_ema+EMA_ALPHA_SIM*float(t_step_s[1])
        transform_overlays_similarity(overlays, s_ema, theta_ema, (tx_ema,ty_ema))
        did_motion=True
    else:
        if USE_ORB_FALLBACK and (prev_gray_full is not None):
            M2=orb_similarity(prev_gray_full, gray)
            if M2 is not None:
                s2,R2,t2s=project_to_similarity(M2)
                s2=max(1.0-MAX_SCALE_STEP, min(1.0+MAX_SCALE_STEP, s2))
                theta2=angle_from_R(R2)
                theta2=max(-math.radians(MAX_ROT_STEP_DEG), min(math.radians(MAX_ROT_STEP_DEG), theta2))
                s_ema=(1-EMA_ALPHA_SIM)*s_ema+EMA_ALPHA_SIM*s2
                theta_ema=(theta_ema+((theta2-theta_ema+math.pi)%(2*math.pi)-math.pi)*EMA_ALPHA_SIM)
                tx_ema=(1-EMA_ALPHA_SIM)*tx_ema+EMA_ALPHA_SIM*float(t2s[0])*FLOW_DS
                ty_ema=(1-EMA_ALPHA_SIM)*ty_ema+EMA_ALPHA_SIM*float(t2s[1])*FLOW_DS
                transform_overlays_similarity(overlays, s_ema, theta_ema, (tx_ema,ty_ema))
                did_motion=True

    if did_motion:
        M2_step=np.array([[math.cos(theta_ema)*s_ema, -math.sin(theta_ema)*s_ema, float(tx_ema)/FLOW_DS],
                          [math.sin(theta_ema)*s_ema,  math.cos(theta_ema)*s_ema, float(ty_ema)/FLOW_DS]], dtype=np.float32)
    else:
        M2_step=np.array([[1,0,0],[0,1,0]], dtype=np.float32)
    if frame_idx>0:
        sim_steps.append((frame_idx-1, frame_idx, np.vstack([M2_step, [0,0,1]]).astype(np.float32)))
        if len(sim_steps)>SIM_HIST_MAX: sim_steps.pop(0)

    # ---- YOLO finger (ASYNC) ----
    now=time.time()

    with mode_lock:
        in_op_or_guide = (mode_state == MODE_OP) or (mode_state == MODE_GUIDE) #보기 모드에서 YOLO가 비동기로 실행 안되게 막음 (8.21)

    if in_op_or_guide: #보기 모드에서 YOLO가 비동기로 실행 안되게 막음(8.21)
        if yolo_in_q.empty():
            try: yolo_in_q.put_nowait(frame_work.copy())
            except queue.Full: pass
        try: det=yolo_out_q.get_nowait()
        except queue.Empty: det=None
    else:
        det=None

    finger_is_fresh=False
    finger_src="NONE"
    yolo_last_conf=None
    klt_draw_pts=None
    yolo_box_count=None

    if isinstance(det, dict):
        xy=det.get('xy')
        if xy is not None:
            fx,fy=int(xy[0]), int(xy[1])
            if last_finger_xy is None:
                filt=np.array([fx,fy], dtype=np.float32)
            else:
                filt=(1-EMA_ALPHA_FINGER)*np.array(last_finger_xy,dtype=np.float32)+EMA_ALPHA_FINGER*np.array([fx,fy],dtype=np.float32)
            last_finger_xy=(int(filt[0]), int(filt[1]))
            finger_last_seen=now; finger_is_fresh=True; finger_src="YOLO"
            yolo_last_conf=float(det.get('conf',0.0))
            if USE_KLT_FALLBACK:
                klt_pts_prev=_klt_seed_ring(last_finger_xy)
                klt_lost_frames=0; #frames_since_reseed=0 삭제(8.20)
        if isinstance(det, dict) and det.get('yolo_in') is not None:
            yolo_last_in=det['yolo_in']
            if YOLO_SHOW_INPUT: cv2.imshow(YOLO_INPUT_WIN, yolo_last_in)
        if YOLO_DEBUG:
            rb=det.get('raw_boxes',[])
            yolo_box_count=len(rb)
            if YOLO_DRAW_ALL:
                for (x1,y1,x2,y2,conf,cls_id) in rb:
                    cv2.rectangle(frame_disp,(int(x1),int(y1)),(int(x2),int(y2)),(0,200,255),1)
                    cv2.putText(frame_disp,f"{conf:.2f}/{cls_id}",(int(x1),max(0,int(y1)-3)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),1,cv2.LINE_AA)
    # 수정5: KLT execution
    if USE_KLT_FALLBACK and not finger_is_fresh and (prev_gray_klt is not None) and (klt_pts_prev is not None):
        klt_xy, klt_pts_next = klt_track_multi(prev_gray_klt, gray_klt, klt_pts_prev, W, H)
        if klt_xy is not None:
            # KLT 추적 성공
            last_finger_xy = klt_xy
            klt_pts_prev = klt_pts_next
            klt_draw_pts = klt_pts_next # 화면 표시용
            
            finger_last_seen = now
            finger_is_fresh = True # KLT가 찾았어도 'fresh'로 간주하여 OCR 등 후속 로직 실행
            finger_src = "KLT"
            klt_lost_frames = 0
        else:
            # KLT 추적 실패
            klt_lost_frames += 1
            if klt_lost_frames > KLT_LOSS_GRACE:
                klt_pts_prev = None # 추적점이 너무 오래되었으므로 초기화
    # 
    # 수정7: KLT 단독 추적 시간제한
    KLT_TIMEOUT_SEC = 1.0
    if finger_src == "YOLO":
        klt_only_start_ts = 0.0  # YOLO가 잡았으면 타이머 리셋
    elif finger_src == "KLT":
        if klt_only_start_ts == 0.0:
            klt_only_start_ts = now  # KLT 추적 시작, 타이머 개시
        # KLT 추적이 1초 이상 지속되면 포인트 무효화
        elif (now - klt_only_start_ts) > KLT_TIMEOUT_SEC:
            last_finger_xy = None # 손가락 좌표 삭제
            finger_is_fresh = False # tts 방지
            klt_pts_prev = None # klt execution 방지
            klt_only_start_ts = 0.0 # 타이머 리셋
            finger_src = "NONE" 
    else: # "NONE"
        klt_only_start_ts = 0.0 # 아무것도 못 잡았으면 타이머 리셋

    # ---- 모드 분기 ----
    with mode_lock:
        mode_now = mode_state

    # 제거1 --- 중복 키 입력 제거 --- 

    # ---- ROI & OCR + 근접 읽기 (OP 모드에서만) ----
    roi=None; protected_boxes=[]; protected_ids=[]
    if mode_now == MODE_OP and finger_is_fresh and (last_finger_xy is not None):
        fx, fy = last_finger_xy
        roi = clamp_rect(int(fx-ROI_W//2), int(fy-ROI_H//2), ROI_W, ROI_H, W, H)
        last_roi = roi
        last_roi_active_until = now + ROI_KEEPALIVE_GRACE_SEC

        rx,ry,rw,rh=roi
        for it in overlays:
            c=poly_center(it['poly'])
            if (rx<=c[0]<=rx+rw) and (ry<=c[1]<=ry+rh):
                it['expiry']=max(it.get('expiry', now), now + BASE_TTL)# 시간 연장 방식 통일 (8.20)

        for it in overlays:
            bx,by,bw,bh=rect_from_poly(it['poly'])
            if fingertip_overlaps_box((fx,fy),(bx,by,bw,bh)):
                protected_boxes.append((bx,by,bw,bh))
                protected_ids.append(it.get('id'))
                it['expiry']=max(it.get('expiry', now), now + BASE_TTL)
                it['pin_until']=now+PIN_GRACE_SEC

        # 근접 읽기(TTS) - OP 모드에서만
        overlap_items=[]
        for it in overlays:
            bx,by,bw,bh=rect_from_poly(it['poly'])
            if fingertip_overlaps_box((fx,fy),(bx,by,bw,bh)):
                overlap_items.append(it)
        near=None; bestd=1e9
        for it in overlap_items:
            c=poly_center(it['poly']); d=np.hypot(c[0]-fx, c[1]-fy)
            if d<bestd: bestd=d; near=it

        if near is not None:
            txt=str(near.get('text','')).strip()
            conf=float(near.get('conf',0.0))
            speak_ok=(conf>=TTS_CONF) or (_has_korean(txt) and (conf>=TTS_CONF_FALLBACK))
            note=""
            low_conf=(_has_korean(txt) and conf<TTS_CONF_FALLBACK) or (not _has_korean(txt) and TTS_CONF)
            if low_conf: note=(note+f" | low-conf({conf:.2f})") if note else f"low-conf({conf:.2f})"
            say_txt=None
            if speak_ok and txt:
                dict_thr=DICT_THRESHOLD_LOWCONF if low_conf else DICT_THRESHOLD
                mapped, sc = map_to_dict_canon(txt, threshold=dict_thr)
                if mapped:
                    say_txt=mapped; note=(note+f" | dict:{sc:.0f}") if note else f"dict:{sc:.0f}"
                elif not STRICT_DICT_ONLY:
                    thr=JAMO_THRESHOLD_LOWCONF if low_conf else JAMO_THRESHOLD
                    fixed, changed = correct_text(txt, threshold=thr)
                    say_txt=fixed if changed else txt
                    if changed: note=(note+" | spellfix") if note else "spellfix"
            if say_txt:
                set_tts_target(say_txt, note=note,force=True); tts_last_seen_target_ts=now #선점발화 (8.21)
            else:
                if STRICT_DICT_ONLY and (speak_ok and txt):
                    note=(note+" | no-dict") if note else "no-dict"
                set_tts_target(None, note=note); tts_current_display=txt
        else:
            if (now - tts_last_seen_target_ts) > TTS_TARGET_STICKY_SEC:
                set_tts_target(None, note="")

        # ---- OCR 스케줄 (OP 모드에서만) ----
        if OCR_ENABLED:
            want_period=BASE_OCR_PERIOD
            roi_labels=[it for it in overlays if (roi[0]<=poly_center(it['poly'])[0]<=roi[0]+roi[2]
                                                  and roi[1]<=poly_center(it['poly'])[1]<=roi[1]+roi[3])]
            roi_moved_fast=(last_roi is None) or (iou(last_roi, roi) < 0.6)
            roi_empty=(len(roi_labels)==0)
            roi_stale=(len(roi_labels)>0 and all((now - it.get('time',now) > STALE_AGE_SEC) or
                                                 (it.get('conf',0)<LOW_CONF_TH) for it in roi_labels))
            if roi_moved_fast or roi_empty or roi_stale:
                want_period=min(want_period, EXTRA_OCR_PERIOD)

            if (now - last_ocr_time) >= want_period and task_q.qsize()==0:
                gx,gy,gw,gh=roi
                g_roi=gray[gy:gy+gh, gx:gx+gw]
                blur_ok=(variance_of_laplacian(g_roi)>=BLUR_VAR_THRESH) or roi_empty
                avg_step=math.hypot(tx_ema, ty_ema)/max(1e-6, FLOW_DS)
                if blur_ok and avg_step>MOTION_GATE_PX: blur_ok=False
                if blur_ok:
                    rects_to_run=[roi]
                    try:
                        task_q.put_nowait({
                            'frame_work': frame_for_ocr.copy(),
                            'rects': rects_to_run,
                            'roi': roi,
                            'frame_idx': frame_idx,
                        })
                        last_ocr_time=now; last_roi=roi
                    except queue.Full:
                        pass

    elif mode_now == MODE_OP and (last_roi is not None) and (now <= last_roi_active_until):
        # YOLO가 잠깐 끊겨도 최근 ROI 내부 항목들의 TTL을 유지/초기화
        rx,ry,rw,rh = last_roi
        for it in overlays:
            c = poly_center(it['poly'])
            if (rx<=c[0]<=rx+rw) and (ry<=c[1]<=ry+rh):
                it['expiry'] = max(it.get('expiry', now), now + BASE_TTL)
    else:
    # INFO 모드에서는 근접 읽기/ROI OCR 모두 비활성화
        if mode_now == MODE_OP:
            pass

    # 손가락이 사라진 뒤에도 근접 읽기가 남아 반복되는 것 방지
    if mode_now == MODE_OP and not finger_is_fresh: #손가락이 사라진 뒤에도 근접 읽기가 남아 반복되는것을 제거 (8.21)
        if (time.time() - tts_last_seen_target_ts) > TTS_TARGET_STICKY_SEC:
            set_tts_target(None, note="")

    # ---- OCR 결과 병합 (OP 모드에서만) ----
    if mode_now == MODE_OP:
        try:
            while True:
                res=result_q.get_nowait()
                if 'dt_ms' in res:
                    if OCR_EMA is None: OCR_EMA=res['dt_ms']
                    else: OCR_EMA=(1-OCR_EMA_ALPHA)*OCR_EMA + OCR_EMA_ALPHA*res['dt_ms']
                if res.get('new_items'):
                    def _T_from_to(a,b):
                        if b<=a: return np.eye(3,dtype=np.float32)
                        T=np.eye(3,dtype=np.float32)
                        for (src,dst,M3) in sim_steps:
                            if a < dst <= b: T = M3 @ T
                        return T
                    T_cap2now=_T_from_to(res.get('frame_idx',frame_idx), frame_idx)
                    def _apply(poly, M3):
                        P=poly.astype(np.float32)
                        return (P @ M3[:2,:2].T) + M3[:2,2]
                    roi_now=_rect_aabb_after_M(res['roi'], T_cap2now, W, H)
                    new_items=[]
                    for ni in res['new_items']:
                        ni['poly']=_apply(ni['poly'], T_cap2now)
                        bx,by,bw,bh=bbox_of_poly(ni['poly'])
                        cx,cy=bx+bw/2, by+bh/2
                        gx,gy,gw,gh=roi_now
                        if gx<=cx<=gx+gw and gy<=cy<=gy+gh:
                            new_items.append(ni)
                    overlays=merge_update_overlays(overlays, new_items, roi_now, now_ts=time.time(),
                                                   iou_th=MERGE_IOU_TH, center_dist_th=MERGE_CENTER_DIST)
        except queue.Empty:
            pass

    # ---- GUIDE MODE tick (OP 모드에서만) ----
    if mode_now == MODE_OP:
        guide_tick(now, last_finger_xy if finger_is_fresh else None, overlays)

    # ---- Prune & render ----
    now2=time.time()
    if (now2-last_prune) >= PRUNE_TIMEOUT_SEC: #prune 주기 변수화 (8.20)
        overlays = dedupe_same_text_overlays(overlays)
        active_roi = None
        if mode_now == MODE_OP:
            if finger_is_fresh and roi is not None:
                active_roi = roi
            elif (last_roi is not None) and (now2 <= last_roi_active_until):
                active_roi = last_roi
        overlays = prune_overlays(overlays, now2, active_roi=active_roi)
        last_prune=now2

    if roi is not None and (mode_now == MODE_OP) and finger_is_fresh and last_finger_xy is not None:
        cv2.rectangle(frame_disp, (roi[0],roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (120,120,255), 1)

    if last_finger_xy is not None:
        color=(0,255,0) if finger_src=="YOLO" else ((255,0,255) if finger_src=="KLT" else (160,160,160))
        cv2.circle(frame_disp, last_finger_xy, 9, color, -1)
        if finger_src=="KLT" and YOLO_DEBUG and klt_draw_pts is not None:
            for p in klt_draw_pts[:60]:
                cv2.circle(frame_disp, (int(p[0,0]), int(p[0,1])), 2, (180,0,180), -1)

    # 안내 목표 강조(OP 모드에서만)
    if (mode_now == MODE_OP) and GUIDE_MODE and GUIDE_TARGET_ITEM is not None:
        highlight_guide_target(frame_disp, GUIDE_TARGET_ITEM)

    # 오버레이 렌더링(OP 모드에서만)
    if mode_now == MODE_OP:
        draw_overlays(frame_disp, overlays, now2)

    if SHOW_TTS_HINT:
        mode_txt = "MODE: OP" if mode_now == MODE_OP else f"MODE: INFO({int(INFO_PERIOD_SEC)}s)"
        l1=f"{mode_txt} | OCR: {'ON' if (OCR_ENABLED and mode_now==MODE_OP) else 'OFF'} | {OCR_ENGINE}"
        if OCR_EMA is not None and mode_now==MODE_OP: l1+=f"  ~{int(OCR_EMA)} ms"
        l1+=f"   TTS: {'ON' if TTS_ENABLE else 'OFF'}"
        src_txt=f"SRC: {finger_src}"
        if finger_src=="YOLO" and yolo_last_conf is not None: src_txt+=f"  conf={yolo_last_conf:.2f}"
        if finger_src=="KLT" and klt_draw_pts is not None: src_txt+=f"  klt_pts={len(klt_draw_pts)}"
        age_ms=int((now - finger_last_seen)*1000.0) if last_finger_xy is not None else -1
        if age_ms>=0: src_txt+=f"  age={age_ms} ms"
        if yolo_box_count is not None: src_txt+=f"  boxes={yolo_box_count}"
        roi_txt=f"ROI: {ROI_W}x{ROI_H}  ([ ] width  ; ' height)"

        # <<< CHANGED: 현재 target 없더라도 마지막 발화 문구를 HUD에 유지
        say_txt = (tts_current_display.strip() or tts_last_spoken_text.strip())
        say_line=f"SAY: {say_txt}" if say_txt else "SAY: (none)"
        if tts_current_note: say_line+=f"  [{tts_current_note}]"

        guide_txt = f"GUIDE: {'ON' if (mode_now == MODE_OP and GUIDE_MODE) else 'OFF'}"
        if (mode_now == MODE_OP) and GUIDE_TARGET:
            guide_txt += f"  target='{GUIDE_TARGET}'"
        if (mode_now == MODE_OP) and GUIDE_TARGET_ITEM is not None:
            cx, cy = map(int, _overlay_center(GUIDE_TARGET_ITEM))
            guide_txt += f"  tgt@({cx},{cy})"

        lines=[guide_txt, l1, src_txt, roi_txt, say_line]

        img_rgb=cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
        pil=Image.fromarray(img_rgb); draw=ImageDraw.Draw(pil)
        font_path=None
        for p in [r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\NanumGothic.ttf",
                  r"C:\Windows\Fonts\NotoSansCJKkr-Regular.otf",
                  "/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.ttc"]:
            if os.path.isfile(p): font_path=p; break
        font=ImageFont.truetype(font_path, 22) if font_path else ImageFont.load_default()

        pad_x,pad_y,gap=10,8,4
        widths=[draw.textlength(s, font=font) for s in lines]
        tw=int(max(widths)) if widths else 0; lh=24
        th=lh*len(lines)+(len(lines)-1)*gap
        x0,y0=8,6
        bg=Image.new("RGBA",(tw+pad_x*2, th+pad_y*2),(0,0,0,180))
        pil.paste(bg,(x0,y0),bg)
        y=y0+pad_y
        for s in lines:
            draw.text((x0+pad_x,y), s, font=font, fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0))
            y+=lh+gap
        frame_disp[:]=cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    disp_scale=min(1.0, DISPLAY_MAX_W/float(W))
    vis=frame_disp if disp_scale==1.0 else cv2.resize(frame_disp, None, fx=disp_scale, fy=disp_scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(WINDOW_NAME, vis)

    key=cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    elif key==ord('o'):
        OCR_ENABLED = not OCR_ENABLED
        drain_queue(task_q)
        last_ocr_time = 0.0 if OCR_ENABLED else time.time()
        print(f"[OCR] {'ENABLED' if OCR_ENABLED else 'DISABLED'}")
    elif key==ord('t'):
        SHOW_TTS_HINT = not SHOW_TTS_HINT
        print(f"[HUD] {'ON' if SHOW_TTS_HINT else 'OFF'}")
    elif key==ord('s'):
        TTS_ENABLE = not TTS_ENABLE
        set_tts_target(None, note="")
        print(f"[TTS] {'ENABLED' if TTS_ENABLE else 'DISABLED'}")
    elif key==ord('y'):
        YOLO_SHOW_INPUT = not YOLO_SHOW_INPUT
        if not YOLO_SHOW_INPUT:
            try: cv2.destroyWindow(YOLO_INPUT_WIN)
            except: pass
        print(f"[YOLO] INPUT PREVIEW {'ON' if YOLO_SHOW_INPUT else 'OFF'}")
    elif key==ord('p'):
        if yolo_last_in is not None:
            os.makedirs('yolo_inputs', exist_ok=True)
            fname=time.strftime("yolo_inputs/%Y%m%d_%H%M%S.png")
            cv2.imwrite(fname, yolo_last_in)
            print(f"[YOLO] saved input preview -> {fname}")
        else:
            print("[YOLO] no input to save yet")
    # --- ROI 크기 조절 ---
    elif key==ord('['):   # width -
        ROI_W=max(MIN_ROI_W, ROI_W-40)
    elif key==ord(']'):   # width +
        ROI_W=min(W, ROI_W+40)
    elif key==ord(';'):   # height -
        ROI_H=max(MIN_ROI_H, ROI_H-30)
    elif key==ord("'"):   # height +
        ROI_H=min(H, ROI_H+30)
    elif key==ord('r'):   # reset ROI
        ROI_W, ROI_H = 420, 420
        print("[ROI] reset to 420x420")

    # ---- 모드 전환 키 ---- ---- 모드 전환 키 ----
    elif key == ord('1'):
        _enter_op_mode()
        print("[MODE] OP")

    elif key == ord('2'):
        _enter_info_mode()
        print("[MODE] INFO")

    elif key == ord('3'):
        _enter_guide_mode()
        print("[MODE] GUIDE")

    # ---- GUIDE mode keys ----
    elif key == ord('c'):
        GUIDE_TARGET = None
        GUIDE_TARGET_ITEM = None
        set_tts_target("목표를 취소했습니다.", note="guide cancel")
        print("[GUIDE] target cleared")

    elif key == ord('v'):
        if mode_state == MODE_GUIDE:
            text = stt_listen_once(timeout=4, phrase_time_limit=4) if (USE_STT and _STT_OK) else None
            if text:
                print(f"[STT] heard: {text}")
                set_guide_target_from_text(text)
            else:
                print("[STT] no text")
        else:
            set_tts_target("먼저 3번을 눌러 안내 모드를 켜 주세요.", note="guide")

    elif key == ord('f'):
        if mode_state == MODE_GUIDE:
            try:
                print("\n[GUIDE] 입력 예시: '세탁', '건조맞춤' ...")
                user_in = input("[GUIDE] 목표 단어 입력: ").strip()
                if user_in:
                    set_guide_target_from_text(user_in)
            except Exception:
                pass
        else:
            set_tts_target("먼저 3번을 눌러 안내 모드를 켜 주세요.", note="guide")


    frame_idx+=1
    prev_gray_full=gray.copy()
    prev_gray_klt=gray_klt.copy()

# cleanup
task_q.put(None)
_info_stop.set()
if TTS_ENABLE and 'tts' in globals() and tts:
    tts_stop.set()
    # tts_q.put(None) 삭제 (8.20)
    try: tts.close()
    except Exception: pass
yolo_stop.set()
cap.release()
cv2.destroyAllWindows()