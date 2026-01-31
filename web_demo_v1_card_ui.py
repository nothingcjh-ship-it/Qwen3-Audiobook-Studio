# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
A gradio demo for Qwen3 TTS models (All-in-One).
Final Polish: Aggressive Light Mode Enforcement.
"""
print("Script started...")

import argparse
import os
import tempfile
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

# --- Global State Management ---
class ModelManager:
    def __init__(self):
        self.tts: Optional[Qwen3TTSModel] = None
        self.ckpt_path: str = ""
        self.model_type: str = "Unknown"
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype: torch.dtype = torch.bfloat16
        self.attn_impl: Optional[str] = None

    def load(self, ckpt: str, device: str, dtype_str: str, use_flash_attn: bool):
        try:
            print(f"Loading model from {ckpt}...", flush=True)
            self.device = device
            self.ckpt_path = ckpt
            
            d_lower = dtype_str.lower()
            if d_lower in ("bf16", "bfloat16"):
                self.dtype = torch.bfloat16
            elif d_lower in ("fp16", "float16", "half"):
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
            
            self.attn_impl = "flash_attention_2" if use_flash_attn else None

            self.tts = Qwen3TTSModel.from_pretrained(
                ckpt,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_impl,
            )
            
            self.model_type = getattr(self.tts.model, "tts_model_type", "base")
            if self.model_type not in ("custom_voice", "voice_design", "base"):
                if "CustomVoice" in ckpt: self.model_type = "custom_voice"
                elif "VoiceDesign" in ckpt: self.model_type = "voice_design"
                else: self.model_type = "base"

            print(f"Model loaded successfully. Type detected: {self.model_type}", flush=True)
            return f"Loaded: {self.model_type} ({ckpt})"
        
        except Exception as e:
            traceback.print_exc()
            return f"Error: {str(e)}"

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

# --- Logic Functions ---

def run_voice_clone(ref_aud, ref_txt, use_xvec, text, lang_disp, t, p, k, r):
    model, err = _check_model_ready("base")
    if err: return None, err
    try:
        if not text: return None, "Text required (请输入文本)."
        at = _audio_to_tuple(ref_aud)
        if at is None: return None, "Audio required (请上传音频)."
        
        supported_langs = getattr(model.model, "get_supported_languages", lambda: ["Auto", "English", "Chinese"])()
        _, lang_map = _build_choices_and_map(supported_langs)
        language = lang_map.get(lang_disp, "Auto")

        wavs, sr = model.generate_voice_clone(
            text=text, language=language, ref_audio=at, ref_text=ref_txt, x_vector_only_mode=use_xvec,
            **_gen_kwargs(t, p, k, r)
        )
        return _wav_to_gradio_audio(wavs[0], sr), "Success (生成成功)."
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {e}"

def save_voice_profile(ref_aud, ref_txt, use_xvec):
    model, err = _check_model_ready("base")
    if err: return None
    try:
        at = _audio_to_tuple(ref_aud)
        if at is None: return None
        items = model.create_voice_clone_prompt(
            ref_audio=at, ref_text=ref_txt, x_vector_only_mode=use_xvec,
        )
        payload = {"items": [asdict(it) for it in items]}
        fd, out_path = tempfile.mkstemp(prefix="voice_profile_", suffix=".pt")
        os.close(fd)
        torch.save(payload, out_path)
        return out_path
    except: return None

def run_voice_design(text, lang_disp, instruct, t, p, k, r):
    model, err = _check_model_ready("voice_design")
    if err: return None, err
    try:
        supported_langs = getattr(model.model, "get_supported_languages", lambda: ["Auto", "English", "Chinese"])()
        _, lang_map = _build_choices_and_map(supported_langs)
        language = lang_map.get(lang_disp, "Auto")
        
        wavs, sr = model.generate_voice_design(
            text=text, language=language, instruct=instruct,
            **_gen_kwargs(t, p, k, r)
        )
        return _wav_to_gradio_audio(wavs[0], sr), "Success."
    except Exception as e: return None, f"Error: {e}"

def run_custom_voice(text, lang_disp, spk_disp, instruct, t, p, k, r):
    model, err = _check_model_ready("custom_voice")
    if err: return None, err
    try:
        supported_langs = getattr(model.model, "get_supported_languages", lambda: ["Auto", "English", "Chinese"])()
        _, lang_map = _build_choices_and_map(supported_langs)
        language = lang_map.get(lang_disp, "Auto")
        wavs, sr = model.generate_custom_voice(
            text=text, language=language, speaker=spk_disp, instruct=instruct,
            **_gen_kwargs(t, p, k, r)
        )
        return _wav_to_gradio_audio(wavs[0], sr), "Success."
    except Exception as e: return None, f"Error: {e}"

# --- UI Builder ---
def build_ui(default_ckpt):
    # Aggressive Light Theme Override
    # We override all standard Gradio color variables to light values
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
    
    /* Force main card to be white */
    .main-card {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 16px !important;
        padding: 30px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02) !important;
    }

    /* Force all groups inside main card to be transparent or white, removing dark grays */
    .gradio-group, .gr-panel { 
        background: transparent !important; 
        border: none !important; 
    }
    
    /* Inputs */
    input, textarea, .gr-input, .dropdown-wrap { 
        background-color: #ffffff !important; 
        border: 1px solid #ddd !important;
        color: #000000 !important;
    }

    /* Sidebar Buttons */
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

    /* Primary Button */
    .primary-btn { 
        background-color: #000000 !important; 
        color: white !important; 
        font-weight: bold !important;
    }
    """
    
    # theme=gr.themes.Base() resets a lot of defaults, making overrides easier
    with gr.Blocks(css=css, title="Qwen3-TTS", theme=gr.themes.Base()) as demo:
        with gr.Row():
            # --- SIDEBAR (Left Column) ---
            with gr.Column(scale=1, min_width=240):
                gr.Markdown(
                    """
                    <div style="margin-bottom: 30px; padding-left: 10px;">
                        <div style="font-size: 20px; font-weight: 900; color: #000;">Qwen3_TTS</div>
                        <div style="font-size: 10px; font-weight: 700; color: #888; letter-spacing: 1px; margin-top: 4px;">NEURAL VOICE ENGINE</div>
                        <div style="font-size: 14px; font-weight: 800; margin-top: 12px; color: #000;">BiuBoom Flow</div>
                        <div style="font-size: 10px; color: #aaa;">Youtube: BiuBoom Flow</div>
                    </div>
                    """
                )
                
                # Navigation Buttons
                nav_custom = gr.Button("👤 Official Roles (官方角色)", elem_classes=["sidebar-btn"])
                nav_design = gr.Button("🎨 Voice Design (语音设计)", elem_classes=["sidebar-btn"])
                nav_clone  = gr.Button("||| Zero-shot Clone (零样本克隆)", elem_classes=["sidebar-btn", "active-btn"]) 
                
                with gr.Accordion("Settings (设置)", open=True):
                    ckpt_input = gr.Textbox(label="Path", value=default_ckpt, show_label=False, placeholder="Model Path...")
                    load_btn = gr.Button("Load Model", size="sm")
                    load_status = gr.Textbox(show_label=False, interactive=False, placeholder="Status...")

                load_btn.click(lambda c: MANAGER.load(c, "mps", "bfloat16", False), inputs=[ckpt_input], outputs=[load_status])

            # --- MAIN CONTENT AREA (Right Column) ---
            with gr.Column(scale=4):
                # Wrapped in a dedicated white card group
                with gr.Group(elem_classes=["main-card"]):
                    
                    # --- VIEW: ZERO-SHOT CLONE ---
                    with gr.Group(visible=True) as view_clone:
                        gr.Markdown("# Zero-shot Clone\n<span style='color:#666;font-size:14px'>Upload audio to clone voice (上传声音进行复刻)</span>")
                        
                        c_tgt_txt = gr.Textbox(label="Text to Synthesize (待合成文本)", lines=3)
                        c_ref_aud = gr.Audio(label="Reference Audio (参考音频)", type="numpy")
                        
                        with gr.Accordion("Advanced Options (展开设置)", open=False):
                            c_ref_txt = gr.Textbox(label="Ref Text (参考文本)", lines=1)
                            c_xvec = gr.Checkbox(label="Use X-Vector Only (仅声纹)", value=False)
                            c_lang = gr.Dropdown(["Auto", "English", "Chinese"], label="Language", value="Auto")
                            ct, cp, ck, cr = [gr.Slider(0.01, 2.0, 0.9, label="Temp"), gr.Slider(0.01, 1.0, 1.0, label="TopP"), gr.Slider(1, 100, 50, label="TopK"), gr.Slider(1.0, 2.0, 1.05, label="RepPen")]
                        
                        c_gen_btn = gr.Button("Execute Clone (执行克隆)", elem_classes=["primary-btn"])
                        
                        gr.Markdown("---")
                        c_out = gr.Audio(label="Result", show_label=False)
                        c_save_btn = gr.Button("Save Voice (保存音色)", size="sm")
                        c_save_out = gr.File(label="Saved Profile")

                        c_gen_btn.click(run_voice_clone, [c_ref_aud, c_ref_txt, c_xvec, c_tgt_txt, c_lang, ct, cp, ck, cr], [c_out])
                        c_save_btn.click(save_voice_profile, [c_ref_aud, c_ref_txt, c_xvec], [c_save_out])

                    # --- VIEW: VOICE DESIGN ---
                    with gr.Group(visible=False) as view_design:
                        gr.Markdown("# Voice Design\n<span style='color:#666;font-size:14px'>Create voice from description (通过描述生成声音)</span>")
                        
                        d_inst = gr.Textbox(label="Voice Description (描述)", lines=2)
                        d_txt = gr.Textbox(label="Text to Synthesize (文本)", lines=3)
                        d_lang = gr.Dropdown(["Auto", "English", "Chinese"], label="Language", value="Auto")
                        
                        d_gen_btn = gr.Button("Generate Voice (生成语音)", elem_classes=["primary-btn"])
                        d_out = gr.Audio(label="Result", show_label=False)

                        d_gen_btn.click(run_voice_design, [d_txt, d_lang, d_inst, ct, cp, ck, cr], [d_out]) 

                    # --- VIEW: OFFICIAL ROLES ---
                    with gr.Group(visible=False) as view_custom:
                        gr.Markdown("# Official Roles\n<span style='color:#666;font-size:14px'>Use high-quality preset voices (使用官方预设音色)</span>")
                        
                        p_spk = gr.Dropdown(["Vivian", "Tech", "News", "CustomerService"], label="Speaker (角色)", value="Vivian")
                        p_txt = gr.Textbox(label="Text to Synthesize (文本)", lines=3)
                        p_lang = gr.Dropdown(["Auto", "English", "Chinese"], label="Language", value="Auto")
                        
                        p_gen_btn = gr.Button("Generate Voice (生成语音)", elem_classes=["primary-btn"])
                        p_out = gr.Audio(label="Result", show_label=False)

                        p_gen_btn.click(run_custom_voice, [p_txt, p_lang, p_spk, d_inst, ct, cp, ck, cr], [p_out])

                # --- Navigation Logic ---
                def show_view(view_name):
                    v_c = gr.update(visible=(view_name == "clone"))
                    v_d = gr.update(visible=(view_name == "design"))
                    v_p = gr.update(visible=(view_name == "custom"))
                    
                    def get_cls(is_active):
                        return ["sidebar-btn", "active-btn"] if is_active else ["sidebar-btn"]
                    
                    b_c = gr.update(elem_classes=get_cls(view_name == "clone"))
                    b_d = gr.update(elem_classes=get_cls(view_name == "design"))
                    b_p = gr.update(elem_classes=get_cls(view_name == "custom"))
                    
                    return [v_c, v_d, v_p, b_c, b_d, b_p]
                
                nav_clone.click(lambda: show_view("clone"), None, [view_clone, view_design, view_custom, nav_clone, nav_design, nav_custom])
                nav_design.click(lambda: show_view("design"), None, [view_clone, view_design, view_custom, nav_clone, nav_design, nav_custom])
                nav_custom.click(lambda: show_view("custom"), None, [view_clone, view_design, view_custom, nav_clone, nav_design, nav_custom])

    return demo

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="", help="Initial model path")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    args, _ = parser.parse_known_args(argv)

    if args.checkpoint:
        MANAGER.load(args.checkpoint, args.device, args.dtype, not args.no_flash_attn)

    demo = build_ui(args.checkpoint)
    print(f"Launching on {args.ip}:{args.port}...")
    demo.queue().launch(server_name=args.ip, server_port=args.port, share=args.share)

if __name__ == "__main__":
    import sys
    sys.exit(main())
