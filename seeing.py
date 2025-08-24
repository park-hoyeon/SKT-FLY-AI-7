# seeing.py
# INFO 모드에서 현재 프레임을 분석해 "요약 문장"을 만들어주는 모듈.
# test (1).py의 최신 로직을 모두 포함하여 재구성되었습니다.

import os
import re
import math
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# ==============================
# 모듈 레벨 설정 및 상수
# ==============================
SIDE_LEFT  = ["통살균", "원격제어", "예약", "내마음"]
SIDE_RIGHT = ["터보샷", "구김방지", "알림음", "빨래추가"]
SIDE_EUCLID_MAX_REL = 0.08

CATEGORY_OPTIONS = {
    "세탁":   ["불림", "애벌세탁", "강력", "표준", "적은때"],
    "헹굼":   ["5회", "4", "3", "2", "1"],
    "탈수":   ["건조맞춤", "강", "중", "약", "섬세"],
    "물온도": ["95", "60", "40", "30", "냉수"],
}
READOUT_ORDER = ("세탁", "헹굼", "탈수", "물온도")

LABEL_SYNONYMS = {
    r"\s+": "",
    r"[＊*()\[\]]": "",
    r"^이?터보\s*샷?$": "터보샷",
    r"\*?터보\s*샷": "터보샷",
    r"\*?알림\s*음(?:\(3초\))?": "알림음",
    r"Wi[\-\s]?Fi": "WiFi",
    r"일회": "1회", r"이회": "2회", r"삼회": "3회", r"사회": "4회", r"오회": "5회",
    r"95\s*℃|95도": "95", r"60\s*℃|60도": "60",
    r"40\s*℃|40도": "40", r"30\s*℃|30도": "30",
}

SIDE_SET = set(SIDE_LEFT + SIDE_RIGHT)
CAT2SET  = {k:set(v) for k,v in CATEGORY_OPTIONS.items()}
ALL_ALLOWED = SIDE_SET.union(*CAT2SET.values())

# --- 중앙 밴드 설정 ---
CENTER_BAND_PAD_REL = 0.06
CENTER_BAND_FALLBACK = (0.34, 0.66)
CENTER_RIGHT_MIN_PX   = 6
CENTER_RIGHT_MIN_FRAC = 0.18

# --- 사이드 매칭 설정 ---
SIDE_COLW_REL  = 0.08
SIDE_DMAX_REL  = 0.25
SIDE_Y_GAP_MIN = 2
SIDE_Y_TOL_REL = 0.02

# ==============================
# 내부 헬퍼 함수들
# ==============================

def _canon_text(raw: str) -> str:
    if not raw: return ""
    s = str(raw)
    for pat, rep in LABEL_SYNONYMS.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    m = re.fullmatch(r"([1-4])회", s)
    if m:
        s = m.group(1)
    elif re.fullmatch(r"5", s):
        s = "5회"
    digits = re.sub(r"[^0-9]", "", s)
    if digits and any(digits in v for v in CATEGORY_OPTIONS.values()):
        s = digits if s != "5회" else "5회"
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    return s

def _is_side_button(tok: str) -> bool:
    return tok in SIDE_SET

def _which_category(tok: str):
    for cat, opts in CAT2SET.items():
        if tok in opts: return cat
    return None

def _order_pts(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

def _warp_points(H, pts_xy):
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1,1,2)
    return cv2.perspectiveTransform(pts, H).reshape(-1,2)

def _map_rect_from_rectified(Hinv, x, y, w, h, offset=(0,0)):
    corners = np.float32([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
    mapped = _warp_points(Hinv, corners)
    x1,y1 = mapped.min(axis=0); x2,y2 = mapped.max(axis=0)
    ox, oy = offset
    return int(x1+ox), int(y1+oy), int(x2-x1), int(y2-y1)

def _easyocr_to_items(detections):
    items = []
    for bbox, text, conf in detections:
        quad = np.array(bbox, dtype=float)
        xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
        cx, cy = float(sum(xs)/4), float(sum(ys)/4)
        xyxy = np.array([min(xs), min(ys), max(xs), max(ys)], dtype=float)
        items.append({"text": text.strip(), "conf": float(conf),
                      "box": quad, "center": (cx, cy), "xyxy": xyxy})
    return items

def _detect_panel_roi(img_bgr, v_pctl=35, bh_kernel=31, min_area_frac=0.08, ar_range=(1.1, 4.0), pad_frac=0.01):
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_kernel, bh_kernel))
    bh = cv2.morphologyEx(V, cv2.MORPH_BLACKHAT, k)
    _, m_bh = cv2.threshold(bh, max(20, bh.mean() + 1.0*bh.std()), 255, cv2.THRESH_BINARY)
    thr_dark = int(np.percentile(V, v_pctl))
    m_dark = cv2.inRange(V, 0, thr_dark)
    mask = cv2.bitwise_or(m_bh, m_dark)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)), 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = h, w
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area_frac * (H*W): continue
        hull = cv2.convexHull(c)
        x,y,wid,hei = cv2.boundingRect(hull)
        ar = max(wid,hei) / max(1, min(wid,hei))
        if not (ar_range[0] <= ar <= ar_range[1]): continue
        if (best is None) or (area > best[0]):
            best = (area, (x,y,wid,hei))
    if best is None:
        return (0,0,W,H), mask
    x,y,wid,hei = best[1]
    pad = int(pad_frac * max(H, W))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + wid + pad); y1 = min(H, y + hei + pad)
    return (x0,y0,x1,y1), mask

def _deskew_panel_by_mask(panel_bgr, panel_mask_roi, min_quad_area_frac=0.05):
    h, w = panel_bgr.shape[:2]
    cnts, _ = cv2.findContours(panel_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return panel_bgr, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < (min_quad_area_frac * h * w):
        return panel_bgr, None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    src = approx.reshape(4,2).astype(np.float32) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    src = _order_pts(src)
    (tl, tr, br, bl) = src
    Wt = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl))); Wt = max(100, Wt)
    Ht = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl))); Ht = max(100, Ht)
    dst = np.array([[0,0],[Wt-1,0],[Wt-1,Ht-1],[0,Ht-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(panel_bgr, H, (Wt, Ht), flags=cv2.INTER_CUBIC)
    return warped, H

def _build_glare_mask(panel_bgr, v_thr=235, s_thr=45, lap_var_thr=25.0, min_area_rel=1e-4, max_area_rel=2e-2, ar_min=3.0, close_ks=5, open_ks=3, dil_ks=3):
    h, w = panel_bgr.shape[:2]
    hsv = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    m_hi = (V >= v_thr) & (S <= s_thr)
    m = (m_hi.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close_ks,close_ks)), 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(open_ks,open_ks)), 1)
    area_img = float(h*w)
    out = np.zeros_like(m, dtype=np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    for i in range(1, num):
        x,y,wid,hei,area = stats[i]
        rel = area/area_img
        if rel < min_area_rel or rel > max_area_rel: continue
        ar = max(wid,hei)/max(1, min(wid,hei))
        if ar < ar_min: continue
        crop = gray[y:y+hei, x:x+wid]
        if cv2.Laplacian(crop, cv2.CV_64F).var() > lap_var_thr: continue
        out[lab==i] = 255
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dil_ks,dil_ks)), 1)
    ratio = out.sum() / 255.0 / area_img
    return out, float(ratio)

def _apply_deglare_toneclip(panel_bgr, glare_mask, ring_px=3, add_v=18):
    hsv = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    dil = cv2.dilate(glare_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ring_px*2+1, ring_px*2+1)), 1)
    ring = cv2.bitwise_and(dil, cv2.bitwise_not(glare_mask))
    if cv2.countNonZero(ring) == 0:
        return panel_bgr
    ring_med = int(np.median(V[ring.astype(bool)]))
    cap = np.clip(ring_med + add_v, 0, 255).astype(np.uint8)
    V2 = V.copy()
    V2[glare_mask.astype(bool)] = np.minimum(V2[glare_mask.astype(bool)], cap)
    return cv2.cvtColor(cv2.merge([H,S,V2]), cv2.COLOR_HSV2BGR)

def _ocr_with_deglare_when_needed(panel_rect_bgr, reader, area_gate=0.002):
    det_orig = reader.readtext(panel_rect_bgr)
    m_gl, ratio = _build_glare_mask(panel_rect_bgr)
    if ratio < area_gate:
        return det_orig
    degl = _apply_deglare_toneclip(panel_rect_bgr, m_gl)
    det_degl = reader.readtext(degl)
    def _score(dets):
        return sum(c for _,_,c in dets) + 0.3*sum(1 for _,t,_ in dets if len(re.sub(r"[^가-힣0-9]","",t))>0)
    return det_degl if _score(det_degl) >= 0.85 * _score(det_orig) else det_orig

def _build_text_mask_from_easyocr(detections, shape_hw, dilate_px=2):
    H, W = shape_hw[:2]
    mask = np.zeros((H, W), np.uint8)
    if not detections: return mask
    polys = [np.array(bbox, dtype=np.int32).reshape(-1, 1, 2) for bbox, _, _ in detections]
    if polys:
        cv2.fillPoly(mask, polys, 255)
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
            mask = cv2.dilate(mask, k, 1)
    return mask

def _auto_led_params_simple(shape, k_frac=0.015, area_lo_frac=1e-5, area_hi_frac=1.5e-3):
    h, w = shape[:2]
    long_side = max(h, w)
    k_auto = int(round(long_side * k_frac))
    if k_auto % 2 == 0: k_auto += 1
    k_auto = max(5, min(k_auto, 31))
    min_area = max(6, int(h * w * area_lo_frac))
    max_area = max(min_area+1, int(h * w * area_hi_frac))
    return k_auto, min_area, max_area

def _detect_leds_glare_core(img_bgr, k=None, sigma=2.3, ring_px=7, ring_v_thr=200, core_s_thr_bg=78, dv_thr_bg=45, strict_aspect=(2.0, 4.2), strict_extent=0.64, strict_solidity=0.80, include_white=False, exclude_mask=None, dv_thr_any=35, min_short_px=10, min_area_abs=40):
    def _masked_mean_median(img_gray, mask_bool):
        vals = img_gray[mask_bool]
        return (float(vals.mean()), float(np.median(vals))) if vals.size > 0 else (0.0, 0.0)
    k_auto, min_area, max_area = _auto_led_params_simple(img_bgr.shape)
    if not k or k <= 0: k = k_auto
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
    Hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(Hsv)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(g_eq, cv2.MORPH_TOPHAT, se)
    m, s = float(tophat.mean()), float(tophat.std())
    _, seed_th = cv2.threshold(tophat, np.clip(m + sigma*s, 40, 240), 255, cv2.THRESH_BINARY)
    _, seed_v  = cv2.threshold(V, 210, 255, cv2.THRESH_BINARY)
    seed = cv2.bitwise_or(seed_th, seed_v)
    m_color = (cv2.inRange(H, 35, 85) | cv2.inRange(H, 90, 140)) & (cv2.inRange(S, 50, 255) & cv2.inRange(V, 160, 255))
    if include_white: m_color |= (cv2.inRange(S, 0, 60) & cv2.inRange(V, 200, 255))
    reinforced = cv2.bitwise_and(seed, cv2.dilate(m_color, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1))
    ratio = (cv2.countNonZero(reinforced) / float(max(1, cv2.countNonZero(seed)))) if cv2.countNonZero(seed)>0 else 0.0
    core = reinforced if ratio >= 0.3 else seed
    if exclude_mask is not None:
        core = cv2.bitwise_and(core, cv2.bitwise_not(exclude_mask))
    core = cv2.medianBlur(core, 3)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)
    num, lab, stats, cents = cv2.connectedComponentsWithStats(core, 8)
    for i in range(1, num):
        if (min_short_px and stats[i,3] < min_short_px) or not (max(min_area, min_area_abs) <= stats[i,4] <= max_area):
            core[lab == i] = 0
    num, lab, stats, cents = cv2.connectedComponentsWithStats(core, 8)
    leds, ring_kernel = [], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px*2+1, ring_px*2+1))
    for i in range(1, num):
        x,y,wid,hei,area = stats[i]
        aspect = max(wid,hei) / max(1, min(wid,hei))
        if aspect > 6.5: continue
        comp_mask = (lab == i)
        dil = cv2.dilate(comp_mask.astype(np.uint8), ring_kernel, 1).astype(bool)
        ring_mask = np.logical_and(dil, np.logical_not(comp_mask))
        core_v_mean, _ = _masked_mean_median(V, comp_mask)
        _, ring_med = _masked_mean_median(V, ring_mask)
        if (core_v_mean - ring_med) < dv_thr_any: continue
        if ring_med >= ring_v_thr:
            cnts, _ = cv2.findContours((comp_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                solidity = cv2.contourArea(c) / max(1.0, cv2.contourArea(cv2.convexHull(c)))
                extent = area / float(max(1, wid*hei))
                if not (strict_aspect[0] <= aspect <= strict_aspect[1]) or extent < strict_extent or solidity < strict_solidity:
                    continue
        leds.append((int(x), int(y), int(wid), int(hei), (float(cents[i][0]), float(cents[i][1])), float(core_v_mean)))
    return leds

def _norm_ko(s: str) -> str:
    return re.sub(r"\s+", "", s or "")

def _find_category_anchors(items):
    anchors = {}
    for it in items:
        raw = it["text"]; norm = _norm_ko(raw)
        for cat in CATEGORY_OPTIONS.keys():
            if cat in norm:
                x1,y1,x2,y2 = it["xyxy"]; h = (y2 - y1); area = (x2 - x1) * h
                prev = anchors.get(cat)
                if not prev or (h > prev.get("_h", -1)) or (h == prev.get("_h", -1) and area > prev.get("_a", -1)):
                    anchors[cat] = {"center": it["center"], "xyxy": it["xyxy"], "_h": h, "_a": area}
    for cat in anchors:
        anchors[cat].pop("_h", None); anchors[cat].pop("_a", None)
    return anchors

def _compute_center_band(items, img_shape):
    H, W = img_shape[:2]
    xs = [x for it in items if any(cat in _norm_ko(it["text"]) for cat in CATEGORY_OPTIONS.keys()) for x in (it["xyxy"][0], it["xyxy"][2])]
    if len(xs) >= 2:
        left  = max(0.0, min(xs) - CENTER_BAND_PAD_REL * W)
        right = min(float(W), max(xs) + CENTER_BAND_PAD_REL * W)
    else:
        left, right = CENTER_BAND_FALLBACK[0] * W, CENTER_BAND_FALLBACK[1] * W
    return float(left), float(right)

def _match_leds_to_texts(items, leds, img_shape, dmax_px=None, rel_gate=1.1, x_orient_eps=4, y_orient_eps=0):
    Hh, Ww = img_shape[:2]
    dmax_px = dmax_px or max(50, int(0.065 * max(Hh, Ww)))
    band_left, band_right = _compute_center_band(items, img_shape)
    side_colw, side_dmax, side_y_tol, side_eucl_max = SIDE_COLW_REL*max(Hh,Ww), SIDE_DMAX_REL*max(Hh,Ww), SIDE_Y_TOL_REL*Hh, SIDE_EUCLID_MAX_REL*max(Hh,Ww)
    choices = []
    for li, (_x,_y,_w,_h,(cx, cy),bright) in enumerate(leds):
        best_cand = None
        for ti, it in enumerate(items):
            tx, ty, tw, th, raw, x1, *_ = it["center"][0], it["center"][1], it["xyxy"][2]-it["xyxy"][0], it["xyxy"][3]-it["xyxy"][1], it["text"], it["xyxy"][0]
            tok = _canon_text(raw)
            if not tok or tok not in ALL_ALLOWED: continue
            dist = 0
            if _is_side_button(tok):
                if not (band_left > cx or cx > band_right) and ty >= cy-side_y_tol and abs(tx-cx) <= max(side_colw, 0.5*tw):
                    dist = max(0.0, ty-cy) + 0.3 * abs(tx-cx)
                    if dist > side_dmax or math.hypot(tx-cx, ty-cy) > side_eucl_max: continue
            else:
                if band_left <= cx <= band_right and band_left <= tx <= band_right and abs(ty-cy) <= max(y_orient_eps, 0.6*th) and x1 >= cx + max(CENTER_RIGHT_MIN_PX, CENTER_RIGHT_MIN_FRAC*tw):
                    dist = math.hypot(tx-cx, ty-cy)
                    if dist > dmax_px: continue
            if dist > 0 and (not best_cand or dist < best_cand[0]):
                best_cand = (dist, ti, tok)
        if best_cand:
            dist, ti, tok = best_cand
            choices.append((dist, li, ti, tok, float(bright), tuple(items[ti]["center"]), (cx,cy)))
    choices.sort(key=lambda x: x[0])
    used_led, used_txt, pairs_led = set(), set(), []
    for d, li, ti, tok, bri, ptxt, pled in choices:
        if li not in used_led and ti not in used_txt:
            used_led.add(li); used_txt.add(ti)
            pairs_led.append((ptxt, pled, tok, li, bri))
    pairs_led.sort(key=lambda p: (int(p[1][1] // 30), p[1][0]))
    return [p[2] for p in pairs_led], pairs_led

def _choose_and_enforce_categories(pairs_led, items, leds, img_shape, cw_rel=0.06, dmax_rel=0.20, fill_default=None):
    H, W = img_shape[:2]; L = max(H, W)
    colw, dmax = cw_rel * L, dmax_rel * L
    picked = {}
    bucket = {cat: [] for cat in CATEGORY_OPTIONS.keys()}
    for _, _, tok, li, bri in pairs_led:
        cat = _which_category(tok)
        if cat: bucket[cat].append((tok, bri, li))
    for cat, arr in bucket.items():
        if arr: picked[cat] = max(arr, key=lambda x: x[1])[0]
    anchors = _find_category_anchors(items)
    for cat in CATEGORY_OPTIONS:
        if cat in picked: continue
        a = anchors.get(cat)
        if a:
            ax, ay = a["center"]
            cand_leds = sorted([ (bri, idx) for idx, (*_, (cx,cy), bri) in enumerate(leds) if abs(cx-ax)<=colw and cy>=ay-2 ], reverse=True)
            if cand_leds:
                led_center = leds[cand_leds[0][1]][4]
                best_tok, best_d = None, dmax
                for it in items:
                    tok = _canon_text(it["text"])
                    if tok in CAT2SET[cat]:
                        tx, ty = it["center"]
                        if abs(tx - ax) <= colw and ty >= ay - 2:
                            d = math.hypot(tx - led_center[0], ty - led_center[1])
                            if d < best_d: best_d, best_tok = d, tok
                picked[cat] = best_tok or (fill_default.get(cat) if fill_default else "미확인")
    return picked

def _compose_readout(cat_map, side_on, order=READOUT_ORDER):
    cat_parts = [f"{k} {cat_map.get(k,'미확인')}" for k in order if k in cat_map]
    cat_sentence = ", ".join(cat_parts)
    side_sentence = " / ".join(side_on) if side_on else ""
    final_parts = [p for p in (cat_sentence, side_sentence) if p]
    return ", ".join(final_parts) if final_parts else "켜진 표시 없음"

# ==============================
# 최종 요약 진입점
# ==============================
def summarize_scene(frame_bgr: np.ndarray, reader, do_pic=True ,debug_font=None,debug_dir: Optional[str]=None) -> str:
    """
    현재 프레임(frame_bgr)을 분석하여, '조작 패널 상태'에 대한 한국어 요약 문장을 반환.
    - test (1).py의 로직을 통합하여 실시간 프레임에 적용.
    """
    try:
        # 1. 패널 ROI 탐지 및 정사영 변환
        (x0,y0,x1,y1), panel_mask_full = _detect_panel_roi(frame_bgr)
        panel_bgr = frame_bgr[y0:y1, x0:x1].copy()
        panel_mask_roi = panel_mask_full[y0:y1, x0:x1].copy()
        panel_rect, H = _deskew_panel_by_mask(panel_bgr, panel_mask_roi)
        Hinv = np.linalg.inv(H) if H is not None else None

        # 2. OCR (필요 시 디글레어 포함)
        result_panel = _ocr_with_deglare_when_needed(panel_rect, reader)
        items_local = _easyocr_to_items(result_panel)

        # 3. 텍스트 마스크 생성 및 LED 탐지
        text_mask_local = _build_text_mask_from_easyocr(result_panel, panel_rect.shape[:2])
        leds_local = _detect_leds_glare_core(
            panel_rect, k=15, sigma=2.0, include_white=True,
            exclude_mask=text_mask_local, dv_thr_any=22, min_short_px=10, min_area_abs=40
        )

        # 4. OCR 및 LED 결과를 원본 이미지 좌표계로 복원
        items = []
        if Hinv is not None:
            for it in items_local:
                mapped = _warp_points(Hinv, it["box"]) + np.array([x0, y0])
                xs, ys = mapped[:,0], mapped[:,1]
                items.append({"text": it["text"], "conf": it["conf"], "box": mapped.tolist(), "center": (xs.mean(), ys.mean()), "xyxy": np.array([xs.min(), ys.min(), xs.max(), ys.max()])})
        else:
            for it in items_local:
                bx = np.array(it["box"]) + np.array([x0, y0])
                xs, ys = bx[:,0], bx[:,1]
                items.append({"text": it["text"], "conf": it["conf"], "box": bx.tolist(), "center": (xs.mean(), ys.mean()), "xyxy": np.array([xs.min(), ys.min(), xs.max(), ys.max()])})

        leds = []
        if Hinv is not None:
            for (x,y,w,h,c,b) in leds_local:
                gx,gy,gw,gh = _map_rect_from_rectified(Hinv, x,y,w,h, offset=(x0,y0))
                gcx, gcy = (_warp_points(Hinv, [c]) + np.array([x0, y0]))[0]
                leds.append((gx,gy,gw,gh, (gcx, gcy), b))
        else:
            for (x,y,w,h,c,b) in leds_local:
                leds.append((x+x0, y+y0, w,h, (c[0]+x0, c[1]+y0), b))

        # 5. LED-텍스트 매칭
        led_tokens, pairs_led = _match_leds_to_texts(items, leds, frame_bgr.shape)

        # 6. 카테고리별 최종 선택 및 문장 생성
        cat_map = _choose_and_enforce_categories(pairs_led, items, leds, frame_bgr.shape)
        side_on = sorted(list(set(tok for _,_,tok,_,_ in pairs_led if _is_side_button(tok))))
        final_text = _compose_readout(cat_map, side_on)

        # (선택) 함수 시그니처에 추가:

        if do_pic:
            try:
                import time as _time
                from PIL import Image, ImageDraw, ImageFont

                out_dir = debug_dir or os.path.join(os.getcwd(), "debug_summaries")
                os.makedirs(out_dir, exist_ok=True)

                vis = frame_bgr.copy()

                # 1) 박스/도형은 OpenCV로 계속 그림 (빠름)
                for it in items:
                    poly = np.array(it["box"], dtype=np.int32)
                    cv2.polylines(vis, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)

                for (x, y, w, h, (cx, cy), bri) in leds:
                    cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (255, 165, 0), 2)
                    cv2.circle(vis, (int(cx), int(cy)), 3, (255, 165, 0), -1)

                # 2) 텍스트는 PIL로 (한글 OK)
                def _pick_kr_font(size=20, font_path=None):
                    cands = [
                        font_path,
                        r"C:\Windows\Fonts\malgun.ttf",
                        r"C:\Windows\Fonts\malgunbd.ttf",
                        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
                        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    ]
                    for p in cands:
                        if p and os.path.exists(p):
                            try:
                                return ImageFont.truetype(p, size)
                            except Exception:
                                pass
                    return ImageFont.load_default()  # 한글 불가 시 경고용으로만 사용

                pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil)
                font = _pick_kr_font(size=20, font_path=debug_font)

                for it in items:
                    x1, y1 = int(it["xyxy"][0]), int(it["xyxy"][1])
                    # OCR 원문 그대로 쓰는 게 읽기 검증에 유리
                    label = f"{it.get('text','')} ({it.get('conf',0.0):.2f})"

                    # 배경박스 + 텍스트
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1 + 2], fill=(0, 0, 0))
                    draw.text((x1 + 4, y1 - th - 4), label, font=font, fill=(255, 255, 255))

                vis = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

                ts = _time.strftime("%Y%m%d_%H%M%S")
                ms = int((_time.time() % 1) * 1000)
                if final_text:
                    out_put_text= f"{final_text} ({ts}_{ms:03d}).png"
                else:
                    out_put_text = f"미확인 상태 ({ts}_{ms:03d}).png"
                out_path = os.path.join(out_dir,out_put_text)
                cv2.imwrite(out_path, vis)
            except Exception as _e:
                print(f"[seeing.summarize_scene] pic save failed: {_e}")


        return final_text or "켜진 표시 없음"

    except Exception as e:
        # traceback.print_exc() # 디버깅 시 주석 해제
        print(f"[seeing.summarize_scene] error: {e}")
        return "현재 상태를 파악하는데 실패했습니다."