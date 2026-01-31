# 🎧 Qwen3-TTS Audiobook Studio (v1.0)
*The Ultimate Local AI Audiobook Producer*

[简体中文](./README_zh.md) | **English**

---

## 🔥 Why Qwen3-TTS Studio?

Most TTS tools are either too simple or too complex. **Qwen3-TTS Studio** is built specifically for **Audiobook Production**, combining high-end AI research with a user-friendly, portable interface.

- **🚀 Dual-Platform Turbo**: One-click start for Mac (M1/M2/M3) and Windows (CUDA).
- **🎭 Pro Casting**: Advanced multi-role orchestration. No more editing audio clips manually.
- **🎨 Creative Control**: Design unique voices from text descriptions.
- **🛡️ 100% Privacy**: Runs entirely on your local hardware. No cloud, no subscription fees.

---

## 🚀 Quick Start

### Method 1: Portable Mode (Recommended) 🎒

#### 🍎 For Mac Users:
1.  **Download & Unzip**: Download this repository.
2.  **Run**: Double-click `launch_studio.command`.
3.  **Turbo Install**: Automatic local environment setup (`./runtime_env`) using our optimized Pip strategy.

#### 🪟 For Windows Users:
1.  **Run**: Double-click `launch_studio_windows.bat`.
2.  **GPU Power**: Automatically detects NVIDIA GPUs for high-speed CUDA inference.

---

## 🌟 Capabilities

### 1. 🎧 Audiobook Production (Studio Tab)
- **Role-Based Synthesis**: Use a standard `Role: Dialogue` format.
- **Map-Reduce Engine**: Efficiently processes long scripts by grouping character lines to minimize model swapping.
- **Director's Tags**: Use `[p=dur]` (e.g., `[p=1.5]`) to insert precise silence segments.

### 2. 🎨 AI Voice Designer
- **Generative TTS**: Describe a voice (e.g., "Deep, raspy male voice with a slight British accent") and hear it instantly.
- **One-Click Save**: Like the voice? Save it to your `./voices` library with one click and use it in your next book.

### 3. 🧬 State-of-the-Art Cloning
- **Zero-Shot**: Clone any voice from a 5-second sample with near-perfect fidelity.

---

## 📖 Script Example
```text
Narrator: The forest was silent, save for the crackle of a distant fire. [p=0.8]
Geralt: I smell trouble.
Jaskier: You always smell trouble, Geralt! [p=0.5]
```

---

## 🤝 Acknowledgements

Special thanks to the **Alibaba Qwen Team** for open-sourcing the incredible **Qwen3-TTS** model. This project would not be possible without their contribution to the AI community.

- **Model Source**: [Qwen3-TTS on Hugging Face](https://huggingface.co/Qwen)

---

## 👤 Author

**Biuboom Flow**
- 📺 **YouTube**: [@BiuboomFlow](https://www.youtube.com/@BiuboomFlow)
- 🚀 Support the project by subscribing for more AI tools!

---

## ⚠️ Disclaimer

1. **For Research Only**: This project is intended for scholarly and research purposes only.
2. **User Responsibility**: Any misuse of synthesized audio (including but not limited to fraud or impersonation) is the sole responsibility of the user.
3. **Copyright**: The Qwen3-TTS model weights are owned by Alibaba Group.

---

## ⚖️ License
Licensed under **Apache 2.0**. See [LICENSE](./LICENSE) for details.
