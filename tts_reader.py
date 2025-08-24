# tts_reader.py  — import해서 쓰는 모듈 버전
import os, time, threading, queue, hashlib
from typing import Optional, Iterable
import pygame
from google.cloud import texttospeech


def _is_korean(s: str) -> bool:
    return any('가' <= ch <= '힣' for ch in (s or ""))


class TTSReader:
    """
    - say(text): 비동기 합성+재생 (메인 루프 non-blocking)
    - 같은 문구 과도 반복 방지(cooldown_sec)
    - 텍스트별 mp3 캐시(tts_cache/)로 재사용
    - 한/영 자동 보이스 선택
    - ignore/min_len로 노이즈 필터 가능
    - credentials_path를 넘기지 않으면 GOOGLE_APPLICATION_CREDENTIALS 환경변수 사용
    """
    def __init__(
        self,
        *,
        credentials_path: Optional[str] = None,
        cache_dir: str = "tts_cache",
        cooldown_sec: float = 1.2,
        speaking_rate: float = 1.05,
        pitch: float = 0.0,
        ko_voice: str = "ko-KR-Standard-A",
        en_voice: str = "en-US-Standard-C",
        min_len: int = 2,
        ignore: Optional[Iterable[str]] = None,
    ):
        # 인증
        if credentials_path:
            self.client = texttospeech.TextToSpeechClient.from_service_account_file(credentials_path)
        else:
            self.client = texttospeech.TextToSpeechClient()

        # 기본 필터
        self.ignore = set(["", None, "None", "hand not detected", "hand detected, but ocr doesn't exist"])
        if ignore:
            self.ignore |= set(ignore)
        self.min_len = min_len

        # 보이스/오디오 설정
        self.ko_voice = ko_voice
        self.en_voice = en_voice
        self.speaking_rate = speaking_rate
        self.pitch = pitch
        self.cooldown_sec = cooldown_sec

        # 캐시
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # 상태
        self.last_text = ""
        self.last_time = 0.0
        self._running = True

        # 재생 스레드
        self.q = queue.Queue()
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        target_fn = getattr(self, '_worker', None)
        if target_fn is None:
            # 안전장치: 동일 로직의 임시 워커 생성
            def target_fn():
                while self._running:
                    text = self.q.get()
                    if text is None:
                        break
                    try:
                        path = self._synth_if_needed(text)
                        self._play(path)
                    except Exception as e:
                        print(f"[TTS] error: {e}")
        self.worker = threading.Thread(target=target_fn, daemon=True)
        self.worker.start()

    # 컨텍스트 매니저 지원 (선택)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        """앱 종료 시 호출(선택)."""
        self._running = False
        self.q.put(None)
        try:
            self.worker.join(timeout=2.0)
        except Exception:
            pass
        # pygame.mixer.quit()  # 앱 전체에서 mixer 공유 시 보통 유지

    # ---------- public API ----------
    def say(self, text: Optional[str]) -> bool:
        """
        텍스트를 읽도록 큐에 추가. 스킵되면 False, 큐에 들어가면 True.
        디바운스/필터/길이 조건을 통과해야 읽음.
        """
        text = (text or "").strip()
# 추가 1: 한국어일 때만 읽기
        if not _is_korean(text):
            return False
        
        if not text or text in self.ignore or len(text) < self.min_len:
            return False

        now = time.time()
        if text == self.last_text and (now - self.last_time) < self.cooldown_sec:
            return False

        self.last_text = text
        self.last_time = now
        self.q.put(text)
        return True

    def say_if_close(self, text: Optional[str], distance: float, threshold: float = 100.0) -> bool:
        """
        손가락-텍스트 거리가 threshold보다 가까울 때만 읽고 싶을 때 사용.
        """
        if distance is None or distance >= threshold:
            return False
        return self.say(text)
# 추가 2: 큐 비우기
    def clear_queue(self):
        """큐에 대기 중인 모든 TTS 요청을 비웁니다."""
        with self.q.mutex:
            self.q.queue.clear()

    # ---------- internals ----------
    def _voice(self, text: str):
        if _is_korean(text):
            return texttospeech.VoiceSelectionParams(language_code="ko-KR", name=self.ko_voice)
        return texttospeech.VoiceSelectionParams(language_code="en-US", name=self.en_voice)

    def _audio_cfg(self):
        return texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=self.speaking_rate,
            pitch=self.pitch,
        )

    def _cache_path(self, text: str) -> str:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.mp3")

    def _synth_if_needed(self, text: str) -> str:
        path = self._cache_path(text)
        if not os.path.exists(path):
            req = texttospeech.SynthesisInput(text=text)
            resp = self.client.synthesize_speech(input=req, voice=self._voice(text), audio_config=self._audio_cfg())
            with open(path, "wb") as f:
                f.write(resp.audio_content)
        return path

    def _play(self, path: str):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and self._running:
            time.sleep(0.03)

    def _worker(self):
        while self._running:
            text = self.q.get()
            if text is None:
                break
            try:
                path = self._synth_if_needed(text)
                self._play(path)
            except Exception as e:
                print(f"[TTS] error: {e}")

    def stop(self):
        try:
            import pygame
            pygame.mixer.music.stop()
        except Exception:
            pass

    def cancel(self):
        try: self.stop()
        except Exception: pass

    def flush(self):
        try: self.stop()
        except Exception: pass

    def is_busy(self):
        try:
            import pygame
            return pygame.mixer.music.get_busy()
        except Exception:
            return False
