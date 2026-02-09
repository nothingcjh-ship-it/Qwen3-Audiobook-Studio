# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
A gradio demo for Qwen3 TTS models (All-in-One).
Final Polish: 9 Official Roles, Smart Availability Check, Auto-Download.
"""
print("Script started...")

import argparse
import os
import tempfile
import traceback
import glob
import json
import gc
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

# --- Constants ---
OFFICIAL_SPEAKERS = [
    {"name": "Vivian", "lang": "Chinese", "desc": "Bright, slightly edgy young female"},
    {"name": "Serena", "lang": "Chinese", "desc": "Warm, gentle young female"},
    {"name": "Uncle_Fu", "lang": "Chinese", "desc": "Seasoned male, low mellow timbre"},
    {"name": "Dylan", "lang": "Chinese (Beijing)", "desc": "Youthful Beijing male, clear"},
    {"name": "Eric", "lang": "Chinese (Sichuan)", "desc": "Lively Chengdu male, husky"},
    {"name": "Ryan", "lang": "English", "desc": "Dynamic male, strong rhythmic"},
    {"name": "Aiden", "lang": "English", "desc": "Sunny American male, clear midrange"},
    {"name": "Ono_Anna", "lang": "Japanese", "desc": "Playful Japanese female"},
    {"name": "Sohee", "lang": "Korean", "desc": "Warm Korean female, rich emotion"},
]

CUSTOM_VOICE_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CUSTOM_VOICE_DIR_NAME = "Qwen3-TTS-12Hz-1.7B-CustomVoice"

FULL_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", 
    "German", "French", "Russian", "Spanish", "Portuguese", "Italian"
]

# --- Global State Management ---
class ModelManager:
    def __init__(self):
        self.tts: Optional[Qwen3TTSModel] = None
        self.ckpt_path: str = ""
        self.model_type: str = "Unknown"
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype: torch.dtype = torch.bfloat16
        self.attn_impl: Optional[str] = None
        self.use_flash_attn = False # Default safely

    def load(self, model_path: str, device: str = "cpu", dtype: str = "auto", use_flash_attn: bool = False):
        try:
            if not model_path or model_path.strip() == "":
                return "Error: No model selected."
            
            print(f"Loading model from {model_path} on {device} (FlashAttn={use_flash_attn})...")
            clean_path = model_path.replace("⚫ ", "").replace("⚪ ", "").strip() # clean status chars
            
            if not os.path.exists(clean_path):
                 return f"Error: Path not found: {clean_path}"

            self.device = device
            self.use_flash_attn = use_flash_attn # Persist this!
            
            if dtype == "auto":
                self.dtype = "auto"
            elif dtype == "float16":
                self.dtype = torch.float16
            elif dtype == "bfloat16":
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32
            
            self.attn_impl = "flash_attention_2" if use_flash_attn else "sdpa" # default to sdpa if not flash
            # On MPS, usually we just let it be None or sdpa equivalent. 
            # Transformers handles it, but passing "flash_attention_2" when not installed throws error.
            if not use_flash_attn: self.attn_impl = None

            self.tts = Qwen3TTSModel.from_pretrained(
                clean_path,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_impl,
            )
            
            self.model_type = getattr(self.tts.model, "tts_model_type", "base")
            if self.model_type not in ("custom_voice", "voice_design", "base"):
                if "CustomVoice" in clean_path: self.model_type = "custom_voice"
                elif "VoiceDesign" in clean_path: self.model_type = "voice_design"
                else: self.model_type = "base"

            print(f"Model loaded successfully. Type detected: {self.model_type}", flush=True)
            return f"Loaded: {self.model_type}"
        
        except Exception as e:
            traceback.print_exc()
            return f"Error: {str(e)}"
    def smart_switch(self, target_type: str):
        """
        Returns (True, "Loaded...") or (False, "Error details...")
        """
        # 1. Check current
        if self.tts is not None and self.model_type == target_type:
            return True, f"Already loaded {target_type}"
            
        debug_msg = [f"Request: {target_type}"]
        
        # 2. Find target model path
        models_dir = os.path.abspath("./models")
        candidates = []
        debug_msg.append(f"Scanning {models_dir}...")
        
        if os.path.exists(models_dir):
            for d in os.listdir(models_dir):
                full = os.path.join(models_dir, d)
                if not os.path.isdir(full): continue
                
                debug_msg.append(f"  Check: {d}")
                
                # Match Logic
                d_normalized = d.lower().replace("_", "").replace("-", "")
                t_normalized = target_type.lower().replace("_", "").replace("-", "")
                
                if t_normalized in d_normalized:
                    candidates.append(full)
                    continue
                    
                # Specific fallbacks
                if target_type == "custom_voice" and "custom" in d.lower(): candidates.append(full)
                if target_type == "base" and "base" in d.lower(): candidates.append(full)
                if target_type == "voice_design" and "design" in d.lower(): candidates.append(full)

        candidates = list(set(candidates))
        
        def sort_key(p):
            s = os.path.basename(p)
            score = 0
            if "1.7B" in s: score += 100
            if "12Hz" in s: score += 10
            return (score, s)
            
        candidates.sort(key=sort_key, reverse=True)
        
        if not candidates:
            return False, "\n".join(debug_msg) + f"\nNo candidates for {target_type}"
            
        target_path = candidates[0]
        debug_msg.append(f"Selected: {target_path}")
        
        # 3. Unload
        if self.tts is not None:
             del self.tts
             self.tts = None
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             if torch.backends.mps.is_available(): torch.mps.empty_cache()
             import gc; gc.collect()
             
        # 4. Load
        # Detect device again
        dev = self.device if self.device else ("cuda" if torch.cuda.is_available() else "mps")
        
        # Use PERSISTED Flash Attn setting
        use_fa = self.use_flash_attn
        
        res = self.load(target_path, dev, "bfloat16", use_fa)
        if "Error" in res:
            return False, f"Load Error: {res}"
            
        return True, f"Switched to {target_path}"
MANAGER = ModelManager()

# --- Helpers ---
def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.replace("_", " ").split()])

def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items: return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping

def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    wav = np.asarray(wav, dtype=np.float32)
    return sr, wav

def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None: return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        return _normalize_audio(wav), int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr
    return None

def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        y = x.astype(np.float32) / max(abs(info.min), info.max) if info.min < 0 else (x.astype(np.float32) - (info.max + 1) / 2.0) / ((info.max + 1) / 2.0)
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6: y = y / (m + eps)
    else: raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip: y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1: y = np.mean(y, axis=-1).astype(np.float32)
    return y

def _check_model_ready(required_type: str = None):
    if MANAGER.tts is None: return None, "Please load a model first (请先加载模型)."
    if required_type and MANAGER.model_type != required_type:
        if required_type == "base" and MANAGER.model_type != "base":
             return None, f"Warning: Loaded model is `{MANAGER.model_type}`, expected `base`."
        if required_type != "base" and MANAGER.model_type != required_type:
            return None, f"Error: Loaded `{MANAGER.model_type}`, but feature needs `{required_type}`."
    return MANAGER.tts, None

def _gen_kwargs(temp, topp, topk, rep_pen):
    return {"temperature": temp, "top_p": topp, "top_k": int(topk), "repetition_penalty": rep_pen, "do_sample": True}

# --- Smart Funcs ---
def smart_split_text(text, max_len=300):
    """
    Splits long text into chunks by punctuation to avoid OOM.
    Prioritizes splitting at: \n, then [. ? ! 。 ？ ！], then [, ，].
    """
    if len(text) < max_len: return [text]
    
    chunks = []
    curr = ""
    
    # Priority 1: Newlines (User's natural paragraphs)
    raw_lines = text.split('\n')
    
    import re
    # Split by sentence endings (Chinese & English)
    # Pattern: keep the delimiter. Includes . ? ! and Chinese variants
    split_pat = r'([。？！.?!])'
    
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        
        # Priority 2: Sentence Endings
        parts = re.split(split_pat, line)
        # Re-attach delimiters
        sentences = []
        for i in range(0, len(parts)-1, 2):
            sent = parts[i] + parts[i+1]
            if sent.strip(): sentences.append(sent)
        
        # Handle last part if no delimiter
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1])
            
        for sent in sentences:
            if len(curr) + len(sent) > max_len:
                if curr.strip(): chunks.append(curr)
                curr = sent
            else:
                curr += sent
                
    if curr.strip(): chunks.append(curr)
    
    # Just in case some chunk is STILL huge (no punctuation), force split
    final_chunks = []
    for c in chunks:
        while len(c) > max_len:
            # Find a comma?
            comma_split = c[:max_len].rfind('，')
            if comma_split == -1: comma_split = c[:max_len].rfind(',')
            
            if comma_split > max_len // 2: # Good split
                final_chunks.append(c[:comma_split+1])
                c = c[comma_split+1:]
            else: # Hard split
                final_chunks.append(c[:max_len])
                c = c[max_len:]
        if c: final_chunks.append(c)
        
    return final_chunks

def generate_long_text_safe(gen_func_callback):
    """
    Wrapper to handle long text segmentation and concatenation.
    callback: function(text_chunk) -> (wav_chunk, sr)
    """
    import gc
    full_wavs = []
    sr = 24000
    
    try:
        # yield progress? We can't easily yielding progress in this helper without passing yield up.
        # But we can assume the caller splits text first.
        # Actually, let's just make the caller loop. 
        pass 
    except: pass
    return None
    
def scan_local_models():
    """Scans for models in ./models."""
    base_dir = "."
    models_dir = os.path.join(base_dir, "models")
    known_models = [
        "Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    ]
    found_choices = []
    
    if os.path.exists(models_dir):
        subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and not d.startswith(".")]
        for d in subdirs:
            full_path = os.path.join(models_dir, d)
            found_choices.append((f"⚫ {d}", full_path))
    
    existing_paths = [c[1] for c in found_choices]
    for km in known_models:
        is_found = False
        for existing in existing_paths:
            if km in existing: is_found = True; break
        if not is_found:
            found_choices.append((f"⚪ {km} (Not Found)", ""))

    # Available first
    found_choices.sort(key=lambda x: (not x[1], x[0])) 
    return found_choices

def check_custom_voice_available():
    """Checks if ANY CustomVoice model is present in ./models."""
    models_dir = "./models"
    if not os.path.exists(models_dir): return False
    for d in os.listdir(models_dir):
        if "CustomVoice" in d and os.path.isdir(os.path.join(models_dir, d)):
            return True
    return False

def get_speaker_choices():
    is_avail = check_custom_voice_available()
    choices = []
    for s in OFFICIAL_SPEAKERS:
        label = s["name"]
        if not is_avail:
            label = f"⚪ {s['name']} (Click to Download Model)"
        else:
             label = f"{s['name']} | {s['lang']}" 
        choices.append(label)
    return choices

def download_custom_voice_model():
    if snapshot_download is None:
        return "Error: huggingface_hub not installed."
    
    target_dir = os.path.join("./models", CUSTOM_VOICE_DIR_NAME)
    print(f"Downloading {CUSTOM_VOICE_REPO} to {target_dir}...")
    try:
        snapshot_download(repo_id=CUSTOM_VOICE_REPO, local_dir=target_dir)
        return f"Successfully downloaded to {target_dir}. Please Click Refresh and Load!"
    except Exception as e:
        return f"Download Failed: {e}"

# --- Audiobook Engine ---
class AudiobookEngine:
    def __init__(self):
        pass

    @staticmethod
    def parse_script(text):
        """
        Parses script text into lines.
        Format: RoleName: Text content... [p=1.0]
        Returns list of dict: {'role': str, 'text': str, 'original': str}
        """
        lines = []
        raw_lines = text.strip().split('\n')
        for l in raw_lines:
            l = l.strip()
            if not l: continue
            
            # Normalization
            l = l.replace("……", "...").replace("——", "，")
            
            role = "Narrator" # Default
            content = l
            
            # Check for Role: format
            if ":" in l:
                parts = l.split(":", 1)
                potential_role = parts[0].strip()
                # Simple heuristic: Role names usually short (< 20 chars)
                if len(potential_role) < 20: 
                    role = potential_role
                    content = parts[1].strip()
            
            lines.append({"role": role, "text": content, "original": l})
        return lines

    @staticmethod
    def _parse_pauses(text, defaults):
        """
        Splits text by [p=x] tags and punctuation.
        Returns list of segments: [{'text': str, 'pause': float}]
        """
        # 1. Extract [p=x] tags
        # Regex for [p=1.2]
        import re
        segments = []
        
        # Split by custom pause tags
        # pattern: (text_before)([p=num])(text_after)
        # We process manually to preserve order
        
        # Pre-process punctuation pauses
        # We don't actually split string by punctuation usually for TTS, 
        # because TTS needs context. 
        # But for "Director's Cut" control, we might need to split sentences 
        # if the user wants specific delays *between* sentences.
        # For MVP: We only apply pauses AFTER the line (Linebreak).
        # AND if the user inserted [p=x] inside the line.
        
        # Let's handle [p=x] first.
        parts = re.split(r'(\[p=[\d\.]+\])', text)
        
        current_text = ""
        for p in parts:
            if p.startswith("[p=") and p.endswith("]"):
                # It's a pause tag
                try:
                    val = float(p[3:-1])
                except: val = 0.5
                segments.append({"text": current_text, "pause": val})
                current_text = ""
            else:
                current_text += p
        
        if current_text:
            segments.append({"text": current_text, "pause": 0.0}) # Last segment has 0 pause (handled by line break later)
            
        return segments

    @staticmethod
    def generate_silence(duration_sec, sr=24000):
        if duration_sec <= 0: return np.array([], dtype=np.float32)
        return np.zeros(int(sr * duration_sec), dtype=np.float32)
    
    @staticmethod
    def normalize_audio(wav, target_rms=0.1):
        """
        Normalize audio to consistent RMS level to prevent volume fluctuations.
        This ensures all chunks have similar loudness when concatenated.
        """
        if len(wav) == 0:
            return wav
        
        # Calculate current RMS
        current_rms = np.sqrt(np.mean(wav**2))
        
        # Avoid division by zero and skip if audio is silent
        if current_rms < 1e-6:
            return wav
        
        # Scale to target RMS
        scaling_factor = target_rms / current_rms
        normalized = wav * scaling_factor
        
        # Prevent clipping
        max_val = np.abs(normalized).max()
        if max_val > 0.95:
            normalized = normalized * (0.95 / max_val)
        
        return normalized

    @staticmethod
    def trim_audio(wav, threshold=0.01):
        """Removes leading/trailing noise/hallucinations."""
        abs_wav = np.abs(wav)
        mask = abs_wav > threshold
        if not np.any(mask): return wav
        start = np.argmax(mask)
        # Give a 50ms buffer after threshold starts to avoid click
        start = max(0, start - int(24000 * 0.05))
        end = len(wav) - np.argmax(mask[::-1])
        end = min(len(wav), end + int(24000 * 0.05))
        return wav[start:end]

    def render(self, script_text, role_map, pause_config, language="Auto", temperature=0.7, top_p=0.8, seed=42, progress=gr.Progress()):
        """
        Render using Grouped Generation Strategy (Map-Reduce).
        """
        import datetime
        import soundfile as sf
        import gc
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(".", "outputs", f"audiobook_{timestamp}")
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run.log")

        def log(msg):
            print(msg)
            with open(log_path, "a", encoding="utf-8") as f: f.write(msg + "\n")
            
        log(f"--- Audiobook Render (Grouped) Started: {timestamp} ---")
        log(f"Language Mode: {language} | Temp: {temperature} | TopP: {top_p} | Seed: {seed}")
        
        # Stability: Set seed for consistent run-to-run results
        if seed is not None:
             torch.manual_seed(seed)
             if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
             if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
        
        # MEMORY FIX: Force garbage collection at start
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
        
        # 1. Parse & Analyze
        parsed = self.parse_script(script_text)
        tasks = [] # List of dict with metadata
        
        for i, line in enumerate(parsed):
            role = line['role']
            text = line['text']
            
            voice = role_map.get(role, role_map.get("Narrator", role_map.get("default", None)))
            if not voice:
                log(f"Warning: No voice for {role}, skipping line {i+1}")
                continue
                
            model_type = "base" if voice.endswith(".pt") else "custom_voice"
            tasks.append({
                "index": i,
                "role": role,
                "text": text,
                "voice": voice,
                "model_type": model_type,
                "audio": None # Placeholder
            })
            
        log(f"Total Tasks: {len(tasks)}")
        
        # 2. Group by Model Type
        groups = {}
        for t in tasks:
            mt = t['model_type']
            if mt not in groups: groups[mt] = []
            groups[mt].append(t)
            
        # 3. Execute Groups
        # Order: CustomVoice first (usually narrators), then Base (clones)
        priority = ["custom_voice", "base"]
        
        for m_type in priority:
            if m_type not in groups: continue
            
            group_tasks = groups[m_type]
            if not group_tasks: continue
            
            log(f"\n>>> Switching to Model: {m_type} for {len(group_tasks)} lines...")
            
            # Smart Switch
            success, switch_msg = MANAGER.smart_switch(m_type)
            log(f"Switch Result: {switch_msg}")
            
            if not success:
                log(f"CRITICAL: Failed to switch. Skipping this group.")
                continue
                
            # Execute Items
            for t_idx, task in enumerate(group_tasks):
                progress((t_idx / len(group_tasks)), desc=f"Generating Group {m_type} ({t_idx+1}/{len(group_tasks)})...")
                
                raw_txt = task['text']
                vc = task['voice']
                
                # Handling segments ([p=x] tags)
                segments = self._parse_pauses(raw_txt, pause_config)
                line_wavs = []
                sr = 24000
                
                try:
                    for seg in segments:
                        raw_txt_seg = seg['text'].strip()
                        if raw_txt_seg:
                            # Buffering: Add leading space to stabilize model start state
                            # This often fixes "filler sounds" or starting pronunciation issues
                            txt = " " + raw_txt_seg 
                            
                            wav = None
                            if m_type == "base":
                                 fpath = os.path.join("./voices", vc)
                                 payload = torch.load(fpath, map_location="cpu")
                                 items = [VoiceClonePromptItem(**x) for x in payload["items"]]
                                 
                                 # MEMORY FIX: No Grad context to prevent graph growth
                                 with torch.no_grad():
                                     w_list, _sr = MANAGER.tts.generate_voice_clone(
                                         text=txt, 
                                         language=language, 
                                         voice_clone_prompt=items,
                                         temperature=temperature, 
                                         top_p=top_p
                                     )
                                 wav = w_list[0]; sr = _sr
                                 del payload, items # Explicit delete
                                 
                            elif m_type == "custom_voice":
                                 spk = vc.split("|")[0].replace("⚫", "").strip()
                                 # MEMORY FIX: No Grad context
                                 with torch.no_grad():
                                     w_list, _sr = MANAGER.tts.generate_custom_voice(
                                         text=txt, 
                                         speaker=spk, 
                                         language=language, 
                                         temperature=temperature, 
                                         top_p=top_p
                                     )
                                 wav = w_list[0]; sr = _sr
                                 
                            if wav is not None:
                                # FIX: Trim hallucinations/fillers at the start
                                wav = self.trim_audio(wav)
                                
                                # FIX: Add tiny silence (0.05s) to avoid syllable clipping
                                pad = self.generate_silence(0.05, sr)
                                line_wavs.append(pad)
                                line_wavs.append(wav)
                        
                        # Add requested pause
                        if seg['pause'] > 0:
                            line_wavs.append(self.generate_silence(seg['pause'], sr))
                    
                    if line_wavs:
                        final_wav = np.concatenate(line_wavs)
                        # Save temp
                        fname = f"seg_{task['index']:03d}.wav"
                        fpath = os.path.join(out_dir, fname)
                        sf.write(fpath, final_wav, sr)
                        task['audio'] = (final_wav, sr)
                        log(f"  -> Generated line {task['index']} ({task['role']})")
                    else:
                        log(f"  -> Skipped line {task['index']}")
                        
                except Exception as e:
                    log(f"  -> Error Line {task['index']}: {e}")
                    import traceback
                    traceback.print_exc()

             # MEMORY FIX: Cleanup after each group loop
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
            # End of Group: Optional cleanup?
            # We let smart_switch handle next cleanup
            
        # 4. Reassemble
        log("\n>>> Reassembling Audio...")
        final_segments = []
        global_sr = 24000
        
        # Sort tasks by index to restore order
        tasks.sort(key=lambda x: x['index'])
        
        for i, task in enumerate(tasks):
            # 1. Content
            if task['audio']:
                wav, sr = task['audio']
                final_segments.append(wav)
                global_sr = sr # Assume consistent SR
            else:
                log(f"Warning: Missing audio for line {i}")
            
            # 2. Tag Pauses inside text? (Already processed in parse? No, we skipped it in simplified render)
            # For MVP Grouped, let's keep it simple: Only Line Pauses.
            # (If we want [p] tags inside, we'd need to split task into sub-tasks. 
            #  Let's assume [p] tags are handled by user splitting lines for now or add basic support later)
            
            # 3. Line Pause
            last_char = task['text'].strip()[-1] if task['text'].strip() else ""
            delay = pause_config.get("newline", 0.5)
            if last_char in ['。', '.']: delay = max(delay, pause_config.get("period", 1.0))
            elif last_char in ['？', '?']: delay = max(delay, pause_config.get("question", 0.6))
            elif last_char in ['，', ',']: delay = max(delay, pause_config.get("comma", 0.3))
            
            if delay > 0:
                final_segments.append(self.generate_silence(delay, global_sr))
                
        if not final_segments:
             err_msg = "Generation Failed. All lines were skipped. Check run.log for 'CRITICAL' errors."
             log(err_msg)
             return None, err_msg
             
        # Check if we have AT LEAST ONE valid audio segment (not just silence)
        has_content = any(len(s) > 24000 for s in final_segments) # silence is usually short? No, pause can be long.
        # Better: check if we generated any tasks successfully.
        
        failed_lines = sum(1 for t in tasks if t['audio'] is None)
        if failed_lines == len(tasks):
             err_msg = "Generation Failed. No speech generated. Check run.log."
             log(err_msg)
             return None, err_msg

        merged = np.concatenate(final_segments)
        final_file = os.path.join(out_dir, "final_audiobook.wav")
        sf.write(final_file, merged, global_sr)
        
        log(f"Done! Saved to {final_file}")
        return (global_sr, merged), f"Success! Audio saved to outputs."

AUDIOBOOK_ENGINE = AudiobookEngine()


def get_all_voice_choices():
    official = get_speaker_choices()
    my_voices = get_my_voices_list()
    # Format official? They are "Name | Lang" or "⚪ Name"
    # Format my voices? "Filename.pt"
    # Return single flat list
    return official + my_voices

def run_audiobook_logic(script, role_state, p_line, p_period, p_comma, p_question, p_hyphen, language, temp, top_p, seed):
    if not script: return None, "Please enter a script."
    if not role_state: return None, "Please configure roles first."
    
    pause_config = {
        "newline": p_line,
        "period": p_period,
        "comma": p_comma,
        "question": p_question,
        "hyphen": p_hyphen
    }
    
    return AUDIOBOOK_ENGINE.render(script, role_state, pause_config, language=language, temperature=temp, top_p=top_p, seed=int(seed))

def add_role_to_state(role, voice, current_state):
    if current_state is None: current_state = {}
    
    # helper for display
    def get_display(d): return [[k, v] for k, v in d.items()]
    
    if not role or not voice: 
        return current_state, get_display(current_state), f"Error: Missing Name or Voice. Current keys: {len(current_state)}"
    
    current_state[role] = voice
    return current_state, get_display(current_state), f"Added: {role} -> {voice}"

PRESETS_FILE = "presets.json"

def get_presets_list():
    if not os.path.exists(PRESETS_FILE): return []
    try:
        with open(PRESETS_FILE, "r", encoding='utf-8') as f:
            data = json.load(f)
            return list(data.keys())
    except:
        return []

def save_preset_logic(name, current_state):
    try:
        if not name or not name.strip(): return "Error: Name cannot be empty.", gr.update()
        if not current_state: return "Error: Cast is empty.", gr.update()
        
        print(f"DEBUG: Saving Preset '{name}' StateType={type(current_state)} Content={current_state}")
        
        data = {}
        if os.path.exists(PRESETS_FILE):
            try:
                with open(PRESETS_FILE, "r", encoding='utf-8') as f: data = json.load(f)
            except Exception as e:
                print(f"Error reading presets: {e}")
                pass
        
        data[name.strip()] = current_state
        
        with open(PRESETS_FILE, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        return f"Saved '{name}'", gr.update(choices=list(data.keys()))
        
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", gr.update()

def load_preset_logic(name):
    if not name: return {}, [], "Please select a preset."
    
    if not os.path.exists(PRESETS_FILE): return {}, [], "No presets file found."
    try:
        with open(PRESETS_FILE, "r", encoding='utf-8') as f:
            data = json.load(f)
            
        state = data.get(name, {})
        display = [[k, v] for k, v in state.items()]
        return state, display, f"Loaded '{name}'"
        
    except Exception as e:
        return {}, [], f"Error: {e}"

def on_select_role(evt: gr.SelectData, data):
    """
    When a user clicks a row in the dataframe, populate the inputs for editing.
    """
    try:
        row_idx = evt.index[0]
        # data is a list of lists: [[role, voice], [role, voice], ...]
        if row_idx < len(data):
            row = data[row_idx]
            return row[0], row[1], f"Editing: {row[0]}"
    except:
        pass
    return gr.update(), gr.update(), ""

# --- Logic Functions ---

def run_voice_clone(ref_aud, ref_txt, use_xvec, text, lang_disp, t, p, k, r):
    model, err = _check_model_ready("base")
    if err: return None, err
    try:
        if not text: return None, "Text required (请输入文本)."
        at = _audio_to_tuple(ref_aud)
        if at is None: return None, "Audio required (请上传音频)."
        
        supported_langs = getattr(model.model, "get_supported_languages", lambda: FULL_LANGUAGES)()
        _, lang_map = _build_choices_and_map(supported_langs)
        language = lang_map.get(lang_disp, "Auto")

        # AUTO-SLICE LOGIC
        chunks = smart_split_text(text, max_len=300)
        print(f"DEBUG: Text length {len(text)}. Sliced into {len(chunks)} chunks.")
        
        full_wav_list = []
        final_sr = 24000
        
        for idx, chunk in enumerate(chunks):
            # SAFETEY CHECK: Skip empty or punctuation-only chunks
            if not any(c.isalnum() for c in chunk):
                print(f"Skipping junk chunk {idx}: '{chunk}'")
                continue
                
            print(f"Processing chunk {idx+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            # MEMORY SAFE CALL
            gc.collect() 
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
            with torch.no_grad():
                wavs, _sr = model.generate_voice_clone(
                    text=chunk, language=language, ref_audio=at, ref_text=ref_txt, x_vector_only_mode=use_xvec,
                    **_gen_kwargs(t, p, k, r)
                )
            
            w = wavs[0]
            # Trim & Normalize & Pad
            w = AudiobookEngine.trim_audio(w)
            w = AudiobookEngine.normalize_audio(w)  # Ensure consistent volume
            w = np.concatenate([w, AudiobookEngine.generate_silence(0.3, _sr)]) # 0.3s pause between chunks
            
            full_wav_list.append(w)
            final_sr = _sr
            
        merged = np.concatenate(full_wav_list)
        return _wav_to_gradio_audio(merged, final_sr), f"Success (Sliced {len(chunks)} parts)."
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {e}"

def save_voice_named(ref_aud, ref_txt, use_xvec, name):
    # 1. Validate inputs first (avoid unnecessary model load)
    at = _audio_to_tuple(ref_aud)
    if at is None: return "Error: Audio required."
    if not name or not name.strip(): return "Error: Name required."
    
    name = "".join(x for x in name if x.isalnum() or x in " _-")

    # 2. Ensure Base model is loaded for prompt creation
    success, msg = MANAGER.smart_switch("base")
    if not success: return f"Error switching to base model: {msg}"
    
    model = MANAGER.tts
    if model is None: return "Error: Model failed to load."
    
    try:
        items = model.create_voice_clone_prompt(
            ref_audio=at, ref_text=ref_txt, x_vector_only_mode=use_xvec,
        )
        payload = {"items": [asdict(it) for it in items]}
        
        out_path = os.path.join("./voices", f"{name}.pt")
        torch.save(payload, out_path)
        return f"Saved to {out_path}"
    except Exception as e: return f"Error: {e}"

def save_designed_voice_logic(audio_info, text, name):
    """
    Saves the generated audio from Voice Design as a new Cloned Voice.
    Uses the generated audio as the reference audio, and the input text as the reference text.
    """
    if audio_info is None: return "Error: No audio generated to save (请先生成语音)."
    if not text: return "Error: Original text required for reference."
    if not name: return "Error: Name required."
    
    # We reuse saving logic from Clone tab
    # save_voice_named handles the audio conversion (and switches to base)
    
    try:
        res = save_voice_named(audio_info, text, False, name)
    finally:
        # Always auto-restore Voice Design model so user can continue designing
        # regardless of save success/failure
        MANAGER.smart_switch("voice_design")
        
    if "Error" not in res:
        return res + "\n(Success! Design Model Reloaded & Ready)"
    
    return res

def get_my_voices_list():
    if not os.path.exists("./voices"): return []
    files = [f for f in os.listdir("./voices") if f.endswith(".pt")]
    files.sort()
    return files

def run_my_voice_logic(text, lang_disp, voice_file, instruct, t, p, k, r):
    model, err = _check_model_ready("base")
    if err: return None, err
    try:
        if not voice_file: return None, "Please select a voice."
        voice_path = os.path.join("./voices", voice_file)
        if not os.path.exists(voice_path): return None, "Voice file not found."
        
        # Load prompt
        payload = torch.load(voice_path, map_location="cpu")
        # Reconstruct items
        # We need VoiceClonePromptItem which is imported
        items = [VoiceClonePromptItem(**x) for x in payload["items"]]
        
        supported_langs = getattr(model.model, "get_supported_languages", lambda: FULL_LANGUAGES)()
        _, lang_map = _build_choices_and_map(supported_langs)
        language = lang_map.get(lang_disp, "Auto")

        # AUTO-SLICE LOGIC
        chunks = smart_split_text(text, max_len=300)
        print(f"DEBUG: MyVoice - Text length {len(text)}. Sliced into {len(chunks)} chunks.")
        
        full_wav_list = []
        final_sr = 24000
        
        for idx, chunk in enumerate(chunks):
            # SAFETEY CHECK: Skip empty or punctuation-only chunks
            if not any(c.isalnum() for c in chunk):
                continue
                
            print(f"Processing chunk {idx+1}/{len(chunks)}...")
            
            # MEMORY SAFE CALL
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
            with torch.no_grad():
                wavs, _sr = model.generate_voice_clone(
                    text=chunk, language=language, voice_clone_prompt=items,
                    **_gen_kwargs(t, p, k, r)
                )
            
            w = wavs[0]
            w = AudiobookEngine.trim_audio(w)
            w = AudiobookEngine.normalize_audio(w)  # Ensure consistent volume
            w = np.concatenate([w, AudiobookEngine.generate_silence(0.3, _sr)])
            
            full_wav_list.append(w)
            final_sr = _sr

        merged = np.concatenate(full_wav_list)
        return _wav_to_gradio_audio(merged, final_sr), f"Success (Sliced {len(chunks)} parts)."
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {e}"

def run_voice_design(text, lang_disp, instruct, t, p, k, r):
    model, err = _check_model_ready("voice_design")
    if err: return None, err
    try:
        supported_langs = getattr(model.model, "get_supported_languages", lambda: FULL_LANGUAGES)()
        _, lang_map = _build_choices_and_map(supported_langs)
        language = lang_map.get(lang_disp, "Auto")
        
        # AUTO-SLICE LOGIC
        chunks = smart_split_text(text, max_len=300)
        full_wav_list = []
        final_sr = 24000
        
        for idx, chunk in enumerate(chunks):
            # SAFETEY CHECK
            if not any(c.isalnum() for c in chunk):
                continue
                
            # Note: Voice Design usually takes "instruct" once. 
            # But we are splitting the target text.
            # We should keep the same instruct for consistency? Yes.
            
            # MEMORY SAFE CALL
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
            with torch.no_grad():
                wavs, _sr = model.generate_voice_design(
                    text=chunk, language=language, instruct=instruct,
                    **_gen_kwargs(t, p, k, r)
                )
            
            w = wavs[0]
            w = AudiobookEngine.trim_audio(w)
            w = AudiobookEngine.normalize_audio(w)  # Ensure consistent volume
            # Short pause for design flow
            w = np.concatenate([w, AudiobookEngine.generate_silence(0.2, _sr)])
            
            full_wav_list.append(w)
            final_sr = _sr
            
        merged = np.concatenate(full_wav_list)
        return _wav_to_gradio_audio(merged, final_sr), f"Success ({len(chunks)} parts)."
    except Exception as e: return None, f"Error: {e}"

def run_custom_voice_logic(text, lang_disp, spk_choice, instruct, t, p, k, r):
    # Check if download needed
    if "Download" in spk_choice:
        yield None, "Downloading Model... This may take a while!"
        msg = download_custom_voice_model()
        yield None, msg
        return

    # Clean name: Handle "Name | Lang" or "⚪ Name ..."
    # We take the first part before any space or separator
    spk_clean = spk_choice.replace("⚪ ", "").split(" ")[0]
    spk_clean = spk_clean.split("|")[0].strip()

    model, err = _check_model_ready("custom_voice")
    if err: 
        yield None, err
        return

    try:
        yield None, "Generating..."
        supported_langs = getattr(model.model, "get_supported_languages", lambda: FULL_LANGUAGES)()
        _, lang_map = _build_choices_and_map(supported_langs)
        language = lang_map.get(lang_disp, "Auto")
        
        # AUTO-SLICE LOGIC
        chunks = smart_split_text(text, max_len=300)
        full_wav_list = []
        final_sr = 24000
        
        for idx, chunk in enumerate(chunks):
            # SAFETEY CHECK
            if not any(c.isalnum() for c in chunk):
                continue
                
            yield None, f"Generating Part {idx+1}/{len(chunks)}..."
            
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
            with torch.no_grad():
                wavs, _sr = model.generate_custom_voice(
                    text=chunk, language=language, speaker=spk_clean, instruct=instruct,
                    **_gen_kwargs(t, p, k, r)
                )
            
            w = wavs[0]
            w = AudiobookEngine.trim_audio(w)
            w = AudiobookEngine.normalize_audio(w)  # Ensure consistent volume
            w = np.concatenate([w, AudiobookEngine.generate_silence(0.3, _sr)])
            
            full_wav_list.append(w)
            final_sr = _sr
            
        merged = np.concatenate(full_wav_list)
        yield _wav_to_gradio_audio(merged, final_sr), f"Success ({len(chunks)} parts)."
    except Exception as e: 
        yield None, f"Error: {e}"

# --- UI Builder ---
def build_ui(default_ckpt):
    # CSS Overrides
    css = """
    :root, .dark, body, .gradio-container { 
        --body-background-fill: #f5f5f5 !important;
        --background-fill-primary: #ffffff !important;
        --background-fill-secondary: #f9f9f9 !important;
        --block-background-fill: #ffffff !important;
        --block-label-background-fill: #ffffff !important;
        --color-accent: #000000 !important;
        --border-color-primary: #e5e5e5 !important;
        --input-background-fill: #ffffff !important;
        --text-color: #000000 !important;
        --body-text-color: #000000 !important;
        --neutral-500: #666 !important;
        --neutral-900: #000 !important;
        background-color: #f5f5f5 !important;
        color: #000000 !important;
    }
    
    /* Force high contrast for labels and text */
    label, span, p, h1, h2, h3, h4, .prose * {
        color: #000000 !important;
    }
    
    /* Make placeholders visible but distinct */
    ::placeholder {
        color: #666666 !important;
        opacity: 1 !important;
    }

    .main-card {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 16px !important;
        padding: 30px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02) !important;
    }

    .gradio-group, .gr-panel { 
        background: transparent !important; 
        border: none !important; 
    }
    
    input, textarea, .gr-input, .dropdown-wrap { 
        background-color: #ffffff !important; 
        border: 1px solid #ddd !important;
        color: #000000 !important;
    }

    .sidebar-btn { 
        color: #666 !important; 
        background: transparent !important;
        justify-content: flex-start !important;
        font-weight: 500 !important;
    }
    .sidebar-btn:hover { 
        background-color: #e0e0e0 !important; 
        color: #000 !important; 
    }
    .active-btn { 
        background-color: #ffffff !important; 
        color: #000000 !important; 
        font-weight: 800 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }

    .primary-btn { 
        background-color: #000000 !important; 
        color: white !important; 
        font-weight: bold !important;
    }
    
    .refresh-btn { 
        min_width: 40px !important; 
        width: 40px !important; 
        padding: 0 !important;
    }
    
    .white-dataframe tbody, .white-dataframe thead, .white-dataframe tr, .white-dataframe td, .white-dataframe th {
        background-color: white !important;
        color: black !important;
    }
    """
    
    with gr.Blocks(css=css, title="Qwen3-TTS", theme=gr.themes.Base()) as demo:
        with gr.Row():
            # --- SIDEBAR ---
            with gr.Column(scale=1, min_width=240):
                gr.Markdown(
                    """
                    <div style="margin-bottom: 30px; padding-left: 10px;">
                        <div style="font-size: 20px; font-weight: 900; color: #000;">Qwen3_TTS</div>
                        <div style="font-size: 10px; font-weight: 700; color: #888; letter-spacing: 1px; margin-top: 4px;">NEURAL VOICE ENGINE</div>
                        <div style="font-size: 14px; font-weight: 800; margin-top: 12px; color: #000;">BiuBoom Flow</div>
                        <div style="font-size: 10px; color: #aaa;">
                            <a href="https://www.youtube.com/@BiuBoomFlow_nothing" target="_blank" style="text-decoration: none; color: #aaa; transition: color 0.3s;">
                                Youtube: BiuBoom Flow
                            </a>
                        </div>
                    </div>
                    """
                )
                
                nav_clone  = gr.Button("||| Zero-shot Clone (零样本克隆)", elem_classes=["sidebar-btn", "active-btn"]) 
                nav_my     = gr.Button("📂 My Voices (我的音色库)", elem_classes=["sidebar-btn"])
                nav_studio = gr.Button("🎧 Audiobook Studio (有声小说)", elem_classes=["sidebar-btn"])
                nav_design = gr.Button("🎨 Voice Design (语音设计)", elem_classes=["sidebar-btn"])
                nav_custom = gr.Button("👤 Official Roles (官方角色)", elem_classes=["sidebar-btn"]) 
                
                with gr.Accordion("Settings (设置)", open=True):
                    # Smart Loader
                    initial_choices = scan_local_models()
                    default_val = ""
                    if default_ckpt and os.path.exists(default_ckpt):
                        for c in initial_choices:
                            if c[1] == default_ckpt: default_val = c[1]; break
                        if not default_val and initial_choices and initial_choices[0][1]: default_val = initial_choices[0][1]

                    ckpt_dropdown = gr.Dropdown(label="Select Model", choices=initial_choices, value=default_val, show_label=False, interactive=True, allow_custom_value=True)
                    
                    with gr.Row():
                        load_btn = gr.Button("Load Model", size="sm")
                        refresh_btn = gr.Button("🔄", size="sm", elem_classes=["refresh-btn"])
                        
                    load_status = gr.Textbox(show_label=False, interactive=False, placeholder="Status...")

                def refresh_list():
                    return gr.update(choices=scan_local_models())

                refresh_btn.click(refresh_list, None, ckpt_dropdown)
                load_btn.click(lambda c: MANAGER.load(c, "mps", "bfloat16", False), inputs=[ckpt_dropdown], outputs=[load_status])

            # --- MAIN CONTENT ---
            with gr.Column(scale=4):
                with gr.Group(elem_classes=["main-card"]):
                    
                    # CLONE
                    with gr.Group(visible=True) as view_clone:
                        gr.Markdown("# Zero-shot Clone\n<span style='color:#666;font-size:14px'>Upload audio to clone voice (上传声音进行复刻)</span>")
                        c_tgt_txt = gr.Textbox(label="Text to Synthesize (待合成文本)", lines=3)
                        c_ref_aud = gr.Audio(label="Reference Audio (参考音频)", type="numpy")
                        with gr.Accordion("Advanced Options (展开设置)", open=False):
                            c_ref_txt = gr.Textbox(label="Ref Text (参考文本)", lines=1)
                            c_xvec = gr.Checkbox(label="Use X-Vector Only (仅声纹)", value=False)
                            c_lang = gr.Dropdown(FULL_LANGUAGES, label="Language", value="Auto")
                            ct, cp, ck, cr = [gr.Slider(0.01, 2.0, 0.9, label="Temp"), gr.Slider(0.01, 1.0, 1.0, label="TopP"), gr.Slider(1, 100, 50, label="TopK"), gr.Slider(1.0, 2.0, 1.05, label="RepPen")]
                        c_gen_btn = gr.Button("Execute Clone (执行克隆)", elem_classes=["primary-btn"])
                        gr.Markdown("---")
                        c_out = gr.Audio(label="Result", show_label=False)
                        c_status = gr.Textbox(visible=True, label="Status (状态)", interactive=False)
                        
                        with gr.Row():
                            c_name = gr.Textbox(label="Voice Name (音色命名)", placeholder="e.g. Boss-01", lines=1)
                            c_save_btn = gr.Button("Save Voice (保存音色)", size="sm")
                        
                        c_save_out = gr.Textbox(label="Save Status", interactive=False)
                        
                        c_gen_btn.click(run_voice_clone, [c_ref_aud, c_ref_txt, c_xvec, c_tgt_txt, c_lang, ct, cp, ck, cr], [c_out, c_status])
                        c_save_btn.click(save_voice_named, [c_ref_aud, c_ref_txt, c_xvec, c_name], [c_save_out])

                    # DESIGN
                    with gr.Group(visible=False) as view_design:
                        gr.Markdown("# Voice Design\n<span style='color:#666;font-size:14px'>Create voice from description (通过描述生成声音)</span>")
                        d_inst = gr.Textbox(label="Voice Description (描述)", lines=2)
                        d_txt = gr.Textbox(label="Text to Synthesize (文本)", lines=3)
                        d_lang = gr.Dropdown(FULL_LANGUAGES, label="Language", value="Auto")
                        d_gen_btn = gr.Button("Generate Voice (生成语音)", elem_classes=["primary-btn"])
                        gr.Markdown("---")
                        d_out = gr.Audio(label="Result", show_label=False)
                        d_status = gr.Textbox(visible=True, label="Status (状态)", interactive=False)
                        
                        with gr.Row():
                            d_name = gr.Textbox(label="Save as New Voice (另存为新音色)", placeholder="e.g. Fantasy-Narrator", lines=1)
                            d_save_btn = gr.Button("Save Voice (保存音色)", size="sm")
                        
                        d_save_out = gr.Textbox(label="Save Status", interactive=False)
                        
                        d_gen_btn.click(run_voice_design, [d_txt, d_lang, d_inst, ct, cp, ck, cr], [d_out, d_status]) 
                        d_save_btn.click(save_designed_voice_logic, [d_out, d_txt, d_name], [d_save_out]) 


                    # MY VOICES (NEW)
                    with gr.Group(visible=False) as view_my:
                        gr.Markdown("# My Voice Library\n<span style='color:#666;font-size:14px'>Manage and use your saved voices (管理和使用保存的音色)</span>")
                        
                        with gr.Row():
                            my_refresh_btn = gr.Button("🔄 Refresh List", size="sm", elem_classes=["refresh-btn"])
                            my_voice_list = gr.Dropdown(label="Select Voice (选择音色)", choices=get_my_voices_list(), interactive=True)
                        
                        my_refresh_btn.click(lambda: gr.update(choices=get_my_voices_list()), outputs=my_voice_list)
                        
                        my_txt = gr.Textbox(label="Text to Synthesize (文本)", lines=3)
                        my_lang = gr.Dropdown(FULL_LANGUAGES, label="Language", value="Auto")
                        
                        my_gen_btn = gr.Button("Generate with My Voice (使用此音色生成)", elem_classes=["primary-btn"])
                        my_out = gr.Audio(label="Result", show_label=False)
                        my_status = gr.Textbox(visible=True, label="Status (状态)", interactive=False)
                        
                        my_gen_btn.click(run_my_voice_logic, [my_txt, my_lang, my_voice_list, d_inst, ct, cp, ck, cr], [my_out, my_status])

                    # OFFICIAL ROLES (UPDATED)
                    with gr.Group(visible=False) as view_custom:
                        gr.Markdown("# Official Roles\n<span style='color:#666;font-size:14px'>Use high-quality preset voices (使用官方预设音色)</span>")
                        
                        # Dynamic choices based on model availability
                        spk_init_choices = get_speaker_choices()
                        p_spk = gr.Dropdown(spk_init_choices, label="Speaker (角色)", value=spk_init_choices[0])
                        
                        p_txt = gr.Textbox(label="Text to Synthesize (文本)", lines=3)
                        p_lang = gr.Dropdown(FULL_LANGUAGES, label="Language", value="Auto")
                        
                        p_gen_btn = gr.Button("Generate Voice (生成语音)", elem_classes=["primary-btn"])
                        p_out = gr.Audio(label="Result", show_label=False)
                        p_status = gr.Textbox(visible=True, label="Status (状态)", interactive=False)

                        p_gen_btn.click(run_custom_voice_logic, [p_txt, p_lang, p_spk, d_inst, ct, cp, ck, cr], [p_out, p_status])


                    # AUDIOBOOK STUDIO
                    with gr.Group(visible=False) as view_studio:
                         gr.Markdown("# Audiobook Studio (有声小说工坊)\n<span style='color:#666;font-size:14px'>Process dialogue with multiple characters (多角色对话生成)</span>")
                         
                         role_state = gr.State({}) # Stores {Role: Voice}
                         
                         # Cast Manager
                         with gr.Accordion("1. Cast Manager (角色管理)", open=True):
                             with gr.Row():
                                 new_role_name = gr.Textbox(label="Role Name (e.g. Narrator)", placeholder="Narrator", scale=2)
                                 new_role_voice = gr.Dropdown(label="Select Voice", choices=[], scale=3, allow_custom_value=True) # Choices filled dynamically
                                 refresh_voices_btn = gr.Button("🔄", scale=0, elem_classes=["refresh-btn"])
                                 add_role_btn = gr.Button("Add/Update", scale=1)
                             
                             role_display = gr.Dataframe(headers=["Role", "Voice"], label="Current Cast", interactive=False, elem_classes=["white-dataframe"])
                             role_log = gr.Textbox(visible=False)
                             
                             def refresh_all_voice_choices(): return gr.update(choices=get_all_voice_choices())
                             refresh_voices_btn.click(refresh_all_voice_choices, outputs=new_role_voice)
                             def refresh_all_voice_choices(): return gr.update(choices=get_all_voice_choices())
                             refresh_voices_btn.click(refresh_all_voice_choices, outputs=new_role_voice)
                             
                             role_display = gr.Dataframe(headers=["Role", "Voice"], label="Current Cast", interactive=False, elem_classes=["white-dataframe"], type="array")
                             role_log = gr.Textbox(visible=False)
                             
                             # Click to Edit
                             role_display.select(on_select_role, [role_display], [new_role_name, new_role_voice, role_log])
                             
                             add_role_btn.click(add_role_to_state, [new_role_name, new_role_voice, role_state], [role_state, role_display, role_log])
                             
                             # Presets UI
                             gr.Markdown("---")
                             with gr.Row():
                                 preset_name = gr.Textbox(label="Save Preset Name", placeholder="My Cast 1", scale=2)
                                 save_preset_btn = gr.Button("💾 Save Cast", scale=1)
                                 
                             with gr.Row():
                                 preset_list = gr.Dropdown(label="Load Preset", choices=get_presets_list(), scale=2)
                                 load_preset_btn = gr.Button("📂 Load Cast", scale=1)
                                 refresh_preset_btn = gr.Button("🔄", scale=0, elem_classes=["refresh-btn"])
                                 
                             preset_status = gr.Textbox(label="Preset Status", interactive=False)
                             
                             # Preset Events
                             save_preset_btn.click(save_preset_logic, [preset_name, role_state], [preset_status, preset_list])
                             load_preset_btn.click(load_preset_logic, [preset_list], [role_state, role_display, preset_status])
                             
                             def refresh_presets(): return gr.update(choices=get_presets_list())
                             refresh_preset_btn.click(refresh_presets, outputs=preset_list)
                         
                         # Pause Control
                         with gr.Accordion("2. Director's Cut (停顿控制)", open=False):
                             gr.Markdown("Set default pause durations (Golden Parameters)")
                             with gr.Row():
                                 p_line = gr.Slider(0.1, 5.0, 0.5, label="Linebreak (换行)")
                                 p_period = gr.Slider(0.1, 5.0, 1.0, label="Period (句号 .)")
                                 p_comma = gr.Slider(0.1, 5.0, 0.3, label="Comma (逗号 ,)")
                                 p_question = gr.Slider(0.1, 5.0, 0.6, label="Question (问号 ?)")
                                 p_hyphen = gr.Slider(0.1, 5.0, 0.3, label="Hyphen (连词符 -)")
                                 
                         # Script
                         gr.Markdown("**3. Script Editor (剧本)**")
                         with gr.Row():
                             # Add Language Selector next to script label or above
                             audiobook_lang = gr.Dropdown(choices=["Auto", "Chinese", "English", "Mix"], value="Auto", label="Global Language (全局语言)", scale=1)
                             script_input = gr.Textbox(label="Script Content (Format: 'Role: Text...')", lines=10, placeholder="Narrator: The story begins.\nHero: Hello world! [p=1.0]\n...", scale=4)
                         
                         gen_book_btn = gr.Button("🎧 Generate Audiobook (开始生成)", elem_classes=["primary-btn"])
                         
                         with gr.Accordion("4. Advanced Settings (高级设置)", open=False):
                             with gr.Row():
                                 param_temp = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Temperature (随机性)", info="Higher = More Emotion/Random")
                                 param_top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.1, label="Top P")
                                 param_seed = gr.Number(value=42, label="Seed (随机种子)", precision=0)
                         
                         book_out = gr.Audio(label="Final Audio", show_label=False)
                         book_status = gr.Textbox(label="Status")
                         
                         gen_book_btn.click(
                             run_audiobook_logic,
                             [script_input, role_state, p_line, p_period, p_comma, p_question, p_hyphen, audiobook_lang, param_temp, param_top_p, param_seed],
                             [book_out, book_status]
                         )

                    # LOGIC ENDS HERE
                    
                # --- Nav Logic ---
                def show_view(view_name):
                    v_c = gr.update(visible=(view_name == "clone"))
                    v_d = gr.update(visible=(view_name == "design"))
                    v_p = gr.update(visible=(view_name == "custom"))
                    v_m = gr.update(visible=(view_name == "myvoices"))
                    v_s = gr.update(visible=(view_name == "studio"))
                    
                    def get_cls(is_active): return ["sidebar-btn", "active-btn"] if is_active else ["sidebar-btn"]
                    b_c = gr.update(elem_classes=get_cls(view_name == "clone"))
                    b_d = gr.update(elem_classes=get_cls(view_name == "design"))
                    b_p = gr.update(elem_classes=get_cls(view_name == "custom"))
                    b_m = gr.update(elem_classes=get_cls(view_name == "myvoices"))
                    b_s = gr.update(elem_classes=get_cls(view_name == "studio"))
                    
                    # Refresh logic
                    d_spk = gr.update()
                    d_studio = gr.update()
                    if view_name == "custom": d_spk = gr.update(choices=get_speaker_choices())
                    if view_name == "studio": d_studio = gr.update(choices=get_all_voice_choices())
                    
                    return [v_c, v_d, v_p, v_m, v_s, b_c, b_d, b_p, b_m, b_s, d_spk, d_studio]
                
                nav_clone.click(lambda: show_view("clone"), None, [view_clone, view_design, view_custom, view_my, view_studio, nav_clone, nav_design, nav_custom, nav_my, nav_studio, p_spk, new_role_voice])
                nav_design.click(lambda: show_view("design"), None, [view_clone, view_design, view_custom, view_my, view_studio, nav_clone, nav_design, nav_custom, nav_my, nav_studio, p_spk, new_role_voice])
                nav_custom.click(lambda: show_view("custom"), None, [view_clone, view_design, view_custom, view_my, view_studio, nav_clone, nav_design, nav_custom, nav_my, nav_studio, p_spk, new_role_voice])
                nav_my.click(lambda: show_view("myvoices"), None, [view_clone, view_design, view_custom, view_my, view_studio, nav_clone, nav_design, nav_custom, nav_my, nav_studio, p_spk, new_role_voice])
                nav_studio.click(lambda: show_view("studio"), None, [view_clone, view_design, view_custom, view_my, view_studio, nav_clone, nav_design, nav_custom, nav_my, nav_studio, p_spk, new_role_voice])

    return demo

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="", help="Initial model path")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    args, _ = parser.parse_known_args(argv)

    if args.checkpoint:
        MANAGER.load(args.checkpoint, args.device, args.dtype, not args.no_flash_attn)

    demo = build_ui(args.checkpoint)
    
    launch_kwargs = {"server_name": args.ip, "share": args.share}
    if args.port is not None:
        launch_kwargs["server_port"] = args.port
        print(f"Launching on {args.ip}:{args.port}...")
    else:
        print(f"Launching on {args.ip} (Auto-Port)...")
        
    demo.queue().launch(**launch_kwargs)

if __name__ == "__main__":
    import sys
    sys.exit(main())
