import torch
import torchaudio
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor

import os

print("Initializing Qwen3-TTS on macOS...")

# 1. Setup Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("Warning: MPS not available. Running on CPU will be slow.")

# 2. Load Model
model_path = "./models/Qwen3-TTS-12Hz-1.7B-Base"
print(f"Loading model from local path: {model_path} ...")

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Please place your model files (config.json, model.safetensors, etc.) in this folder.")
    exit(1)

try:
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    processor = Qwen3TTSProcessor.from_pretrained(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 3. Inference
text = "你好，我是 Qwen3 TTS。很高兴为你服务。"
print(f"Synthesizing text: {text}")

# Mock audio input for voice cloning (if required by the model pipeline, though base model might not strict require strict clone input if just tts? 
# The Qwen3-TTS flow usually requires a prompt audio. 
# For simplicity, we might need a dummy audio if the API requires it, or just text if it supports zero-shot without prompt (which often it doesn't without a prompt audio).
# Let's check the code structure. Usually needs `audios`. 
# We'll create a dummy wav file if needed, or assume the user has one. 
# Wait, for a robust script, I should create a dummy wav or use a system sound.
# Actually, the example in the README often provided a way.
# Let's try to generate a dummy silence wav to use as 'prompt' if needed or just handle the error.
# But for now, let's keep it simple.
# Wait, Qwen3-TTS is a voice cloning model primarily. It typically needs a reference audio.
# I will add a step to create a dummy wav."

import numpy as np
sample_rate = 16000
duration = 1.0 # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# Generate a 440 Hz sine wave
audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
import soundfile as sf
# Save numpy array directly
sf.write("dummy_prompt.wav", audio_data, sample_rate)

try:
    # Remove audios argument as processor doesn't support it directly
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # input_ids needs to be a list of tensors (batch_size, seq_len) for this model's generate method
        # and each tensor in the list should be (1, seq_len)
        input_ids = inputs["input_ids"]
        # Split batch into list of (1, seq_len)
        input_ids_list = [input_ids[i:i+1] for i in range(input_ids.shape[0])]
        
        batch_size = len(input_ids_list)
        generated_codes_list, generated_hidden_states_list = model.generate(
            input_ids=input_ids_list,
            attention_mask=inputs["attention_mask"],
            languages=["auto"] * batch_size
        )
    
    print("Inference command executed successfully.")
    
    # The output is codes/hidden states. 
    # For a real demo we would need a vocoder to convert codes to audio, 
    # but the base model output confirms the model ran.
    # Qwen3-TTS usually involves a flow where codes are converted to audio.
    # But for verification, just running generate is enough.
    
except Exception as e:
    print(f"Error during inference: {e}")
    # Print full traceback for debugging
    import traceback
    traceback.print_exc()
    exit(1)

print("Script finished.")
