
# Helper for saving designed voice
def save_designed_voice_logic(audio_info, text, name):
    if audio_info is None: return "Error: No audio generated to save."
    if not text: return "Error: Original text required for reference."
    if not name: return "Error: Name required."
    
    # audio_info from Gradio is (sr, wav) tuple or filepath? 
    # For gr.Audio(type="filepath") it is a path.
    # For gr.Audio(type="numpy") it is (sr, wav). 
    # The existing code uses type="numpy" default (implied) or checks.
    # Let's check `run_voice_design` return type.
    # It returns `_wav_to_gradio_audio` which usually returns (sr, wav).
    
    # We can reuse `save_voice_named` but we need to handle the input format.
    # `save_voice_named` expects `ref_aud` which `_audio_to_tuple` parses.
    
    return save_voice_named(audio_info, text, False, name)
