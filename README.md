# AI-Director: Cinematic Frame Architect

AI-Director is a specialized orchestration layer designed to bridge the gap between creative prose and consistent video generation. It transforms raw user descriptions into structured, high-fidelity production briefs ready for video synthesis.


#### Demo version limits
The demo version available at eksperymentyprzemka.pl has a budget limit of $5. Please avoid overusing it, as it may stop working once the limit is reached.
For budget updates or additional quota, please contact me directly.

---

## How the Agent Works

The AI-Director operates as a **Structured Reasoning Engine** rather than a simple chatbot. It follows a multi-step cognitive process:

1.  **Deconstruction:** It strips the user's natural language prompt into core visual components (Subject, Setting, Action).
2.  **Schema Enforcement:** Using a "system-first" instruction set, it forces all creative output into a rigid **JSON schema**. This ensures that downstream video models (like Veo, Sora, or Luma) receive machine-readable, error-free data.
3.  **Sanitization:** User input is encapsulated within `<scene_description>` tags, acting as a sandbox to prevent prompt injection and ensure the "Director" persona remains in control of the output format.

## Scene & Character Consistency

The primary challenge in AI video is "flicker" or character drifting. This solution solves this through **Anchor Extraction**:

* **Subject Anchor:** Before generating frames, the agent defines a "Fixed Appearance" profile. This includes specific physical traits (hair color, clothing texture, height) that are repeated in every frame prompt to maintain identity.
* **Environment Anchor:** The agent locks the background setting (lighting, architecture, time of day) into a separate constant. 
* **Prompt Concatenation:** Every individual keyframe prompt is programmatically built by merging the `subject_anchor` + `environment_anchor` + `frame_specific_action`. This ensures the foundation of the image remains stable while only the movement evolves.



## Design Assumptions

During development, the following architectural choices were made:

* **Instruction Isolation:** Assumes that system instructions and user input must be strictly separated to prevent "jailbreaking" of the cinematic style.
* **JSON-Centric Output:** Designed with the assumption that this data will be consumed by an API or an automated video pipeline, requiring valid, parsable code rather than conversational text.
* **Light-Mode UI:** The interface is locked to a light theme via `.streamlit/config.toml` to ensure a consistent, professional "Studio" aesthetic for all users regardless of system settings.
* **Declarative Camera Logic:** Assumes that "Director-level" language (e.g., *Tracking Shot, Low Angle, Rack Focus*) provides better results in video models than generic descriptions.

## Scaling & Future Development

This architecture is designed for modular expansion and high-performance production.

### 1. Advanced Consistency Anchors
A "Pro" version would introduce granular UI controls for technical film parameters:
* **Lens Anchor:** Force specific focal lengths (e.g., 35mm wide-angle vs. 85mm portrait) to maintain consistent spatial compression.
* **Color Palette Anchor:** Lock specific hex codes or digital film stocks (e.g., "Kodak Portra 400").
* **Emotion Anchor:** Ensure the character's mood evolves logically (e.g., "Stoic" → "Surprised" → "Relieved").

### 2. High-Performance Infrastructure
* **Parallel Rendering:** Transition to **Asynchronous Execution (`asyncio`)** to render all  frames simultaneously rather than sequentially, significantly reducing generation latency.
* **Video Generation:** Directly feed static frames into **Runway Gen-3** or **Luma Dream Machine** as strict reference frames (Image-to-Video).

### 3. Upgrading Quality (Paid Tier)
* **Director Agent:** Upgrade to **Gemini 1.5 Pro** or **GPT-4o** for deeper narrative nuance and complex script formatting.
* **Cinematographer Agent:** Integrate **Flux.1 Pro**, **Midjourney v6**, or **DALL-E 3** for photorealistic textures and superior anatomical rendering (hands/faces).

### 4. Custom Fine-Tuning
Training a lightweight "Director" model specifically on professional movie scripts and technical shot lists to master niche terminology like "Dutch Angles" or "Dolly Zooms" with zero-shot accuracy.

---

### 🛠️ Quick Start

1.  **Clone the repo:** `git clone https://github.com/pniedziela/AI-Director.git`
2.  **Install dependencies:** `pip install -r requirements.txt`
3.  **Add your API Key:** Set `google_api_key`, `hf_token` and `access_code` in your environment variables (bash).
```bash
export google_api_key="your_gemini_api_key"
export hf_token="your_huggingface_token"
export access_code="your_app_password"
```
4.  **Launch:** `streamlit run app.py`
