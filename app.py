import streamlit as st
import os
import time
import json
from typing import List, Optional
from enum import Enum
import json_repair
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from huggingface_hub import InferenceClient
import base64


def get_base64_of_bin_file(bin_file):
    """
    Reads a binary file and returns the base64 string.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def add_login_background():
    """
    Adds a background image/style ONLY to the login screen.
    """
    bin_str = get_base64_of_bin_file("unnamed.jpg")
    
    st.markdown(
        f"""
        <style>
        /* Target the main app container */
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bin_str}");
            background-attachment: fixed;
            background-size: cover;
        }}
        
        /* Style the input box to make it readable */
        .stTextInput > label {{
            color: white !important;
            font-size: 20px !important;
            font-weight: bold;
        }}
        
        /* Add a semi-transparent box behind the input */
        div[data-testid="stVerticalBlock"] > div:has(input) {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #00ffcc; /* Neon border */
            max-width: 500px;
            margin: 0 auto; /* Center it */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def remove_login_background():
    """
    Resets the background to default once logged in.
    """
    st.markdown(
        """
        <style>
        .stApp {
            background-image: none;
            background-color: #ffffff; /* Default Streamlit Dark Mode color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def check_password():
    """Returns `True` if the user had the correct password."""

    # 1. Check if we have a password set in secrets.toml
    if "access_code" not in st.secrets:
        st.error("formatting error: 'access_code' is missing from .streamlit/secrets.toml")
        return False

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["access_code"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't keep password in session state
        else:
            st.session_state["password_correct"] = False

    # 2. Initialize Session State
    if "password_correct" not in st.session_state:
        add_login_background()

        # Centered Logo/Title for Login
        st.markdown("<h1 style='text-align: center; color: white;'>Classified Access</h1>", unsafe_allow_html=True)

        # First run, show input
        st.text_input(
            "Enter Access Code", type="password", on_change=password_entered, key="password"
        )
        return False
    
    # 3. Return status
    elif not st.session_state["password_correct"]:
        add_login_background()
        
        st.markdown("<h1 style='text-align: center; color: white;'>Classified Access</h1>", unsafe_allow_html=True)

        # Password incorrect, show input again + error
        st.text_input(
            "Enter Access Code", type="password", on_change=password_entered, key="password"
        )
        st.error(":closed_lock_with_key: Password incorrect")
        return False
    else:
        # Password correct
        remove_login_background()
        return True

if not check_password():
    st.stop()  # STOPS EXECUTION HERE if not logged in

class VisualStyle(str, Enum):
    CINEMATIC = "Cinematic lighting, photorealistic, 8k, shot on 35mm film, Arri Alexa"
    ANIME = "Studio Ghibli style, vibrant colors, detailed background art, cel shaded"
    CYBERPUNK = "Cyberpunk aesthetic, neon lighting, rain-slicked streets, high contrast"
    VINTAGE = "1980s VHS footage, grainy, slight chromatic aberration, retro aesthetic"

class ScenePlan(BaseModel):
    subject_anchor: str = Field(..., description="Fixed description of character/clothing")
    environment_anchor: str = Field(..., description="Fixed description of background")
    lighting_anchor: str = Field(..., description="Fixed lighting description")
    keyframes: List[dict] = Field(..., description="List of frame prompts")

class VideoBackend:
    def __init__(self):
        # LOAD KEYS FROM SECRETS
        self.google_key = st.secrets.get("google_api_key", "")
        self.hf_token = st.secrets.get("hf_token", "")
        
        if self.google_key:
            # FIX: Force 'v1alpha' to access all models including aliases
            self.gemini = genai.Client(
                api_key=self.google_key, 
                http_options={'api_version': 'v1alpha'}
            )
        if self.hf_token:
            self.flux = InferenceClient(token=self.hf_token)

    def plan_scene(self, user_input: str, style: str, frame_count: int) -> ScenePlan:
        if not self.google_key:
            raise ValueError("Google API Key missing in secrets.toml")

        sys_instruction = f"""
                You are an expert Video Director.
                
                # Task:
                1. Extract a "Character Anchor" (fixed appearance).
                2. Extract an "Environment Anchor" (fixed background).
                3. Create {frame_count} sequential keyframe prompts.
                
                # Output valid JSON with keys: "subject_anchor", "environment_anchor", "lighting_anchor", "keyframes".
                # "keyframes" list items must have: "frame_id", "camera", "action", "full_prompt".
                """

        user_prompt = f"""
                Target Visual Style: {style}
                
                <scene_description>
                {user_input}
                </scene_description>
                """

        try:
            # FIX: Use Gemini 2.0 Flash (Experimental) - It is much more robust on the new SDK
            response = self.gemini.models.generate_content(
                        model="models/gemini-2.5-flash", 
                        contents=user_prompt,
                        config={
                            'system_instruction': sys_instruction,  # <--- SECURE SEPARATION
                            'response_mime_type': 'application/json' 
                        }
                    )

            if not response.candidates or not response.candidates[0].content.parts:
                    # Check WHY it refused
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason == "SAFETY":
                        raise ValueError("The AI refused this request due to safety guidelines (Hard Filter).")
                    else:
                        raise ValueError(f"Generation failed. Reason: {finish_reason}")

            # print(response.text)
            # data = json.loads(response.text)
            data = json_repair.loads(response.text)
            return ScenePlan(**data)
        except Exception as e:
            raise RuntimeError(f"Planning failed on  models. Error: {str(e)}")

    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def render_frame_to_base64(self, prompt: str):
        import io
        import base64
        from PIL import Image
        
        try:
            # 1. Generate the image
            # Removed width/height/guidance to ensure basic compatibility first
            image = self.flux.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-schnell"
            )
            
            if image is None:
                raise ValueError("Flux returned an empty response")

            # 2. Convert to Bytes
            buffered = io.BytesIO()
            # Ensure we are saving as RGB to avoid issues with transparency (PNG vs JPEG)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
                
            image.save(buffered, format="JPEG", quality=85)
            
            # 3. Encode
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str

        except TypeError as te:
            # If Flux client is failing on arguments, this will catch it
            raise RuntimeError(f"Argument error in Flux call: {te}")
        except Exception as e:
            raise RuntimeError(f"Base64 Rendering failed: {str(e)}")

def main():
    st.set_page_config(page_title="AI Director Pro", page_icon="🎬", layout="wide")
    
    # Simple Logout Button
    with st.sidebar:
        if st.button("Logout"):
            st.session_state["password_correct"] = False
            st.rerun()
        
        st.divider()
        st.header(":abacus: Settings")
        
        selected_style_name = st.selectbox("Visual Style", [e.name for e in VisualStyle])
        selected_style_prompt = VisualStyle[selected_style_name].value
        frame_count = st.slider("Frame Count", 4, 15, 6)

    st.title(":movie_camera: AI Video Scene Planner")
# --- PROJECT DESCRIPTION ---
    with st.expander(":information_source: How this Agent Works"):
        st.markdown("""
        ### 1. The Core Logic (Agentic Workflow)
        This application uses a **two-stage AI pipeline** to solve the biggest problem in generative video: **Consistency**.
        
        * **Stage 1: The Director (Planner)**
            * The app uses **Google Gemini 2.5 Flash** (via `google-genai` SDK) to act as a creative director.
            * Instead of writing random prompts, it analyzes your request to extract **"Consistency Anchors"**-fixed descriptions of the subject, environment, and lighting that must remain 100% identical across every frame.
            * It then writes a shot-by-shot script, injecting these anchors into every single prompt.
            
        * **Stage 2: The Cinematographer (Renderer)**
            * The app uses **Flux.1-Schnell** (via Hugging Face Inference API) to render the keyframes.
            * This model was chosen for its high prompt adherence and speed, making it ideal for a real-time POC (Proof of Concept).
        
        ### 2. Ensuring Consistency
        Standard AI video tools often "hallucinate" different clothes or backgrounds in every shot. Here it's solved by:
        * **Anchor Locking:** It forces the LLM to separate "Variable Action" (what changes) from "Static Assets" (what stays the same).
        * **Structured JSON:** By enforcing a strict JSON schema, ensuring the output is machine-readable and free of "chatty" AI noise.
        
        ### 3. Design Assumptions (Minimal POC)
        * **Zero-Cost Stack:** This MVP runs entirely on free-tier (I really hope I selected only free models :laughing:) APIs (Google Gemini Free Tier + Hugging Face Free Inference) .
        * **Statelessness:** We use Streamlit's Session State to hold data during the user session, but we do not require a heavy backend database (except for a lightweight local SQLite quota tracker).
        * **Privacy:** The "Gatekeeper" system ensures only authorized users with the access code can trigger the API calls.
        
        ### 4. Future Scalability
        This architecture is ready to scale:
        * **Video Generation:** The static frames can be fed into **Runway Gen-3** or **Luma Dream Machine** as strict reference frames (Image-to-Video).
        * **Parallel Rendering:** We could use `asyncio` to render all 6 frames simultaneously rather than sequentially.
        * **Custom Models:** The "Director" could be fine-tuned on movie scripts to understand cinematic terminology (e.g., "Dutch Angle", "Dolly Zoom") even better.
        * **Upgrading Quality (Paid Tier):**
                    * **Director:** Switch to **Gemini 1.5 Pro** or **GPT-4o** for deeper narrative nuance and script formatting.
                    * **Cinematographer:** Switch to **Flux.1 Pro**, **Midjourney v6 (API)**, or **DALL-E 3** for photorealistic textures and better hand/face rendering.
                    
        * **Advanced Consistency Anchors:**
            * Currently, we track *Subject*, *Environment*, and *Lighting*.
            * A "Pro" version would add UI controls for:
                * **Lens Anchor:** Force specific focal lengths (e.g., 35mm vs 85mm) across all shots.
                * **Color Palette Anchor:** Lock specific hex codes or film stocks (e.g., "Kodak Portra 400").
                * **Emotion Anchor:** Ensure the character's mood evolves logically (e.g., "Stoic" -> "Surprised" -> "Relieved").
                * **Or whatever we feel is needed :** :relaxed:
                """)
    st.markdown("Turn a simple idea into a consistent storyboard.")

    # Initialize Backend with secrets
    # We use try/except to handle missing keys gracefully in the UI
    try:
        backend = VideoBackend()
    except Exception:
        st.error("⚠️ Backend not initialized. Check secrets.toml")
        backend = None

    # --- SESSION STATE ---
    if 'plan' not in st.session_state:
        st.session_state.plan = None
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = {}

    # --- INPUT ---
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area("Scene Description:", value="A cyberpunk samurai in neon rain", height=100)
    with col2:
        st.write("")
        st.write("")
        if st.button("📝 1. Generate Plan", type="primary", use_container_width=True):
            if backend:
                with st.spinner("🤖 Analyzing scene..."):
                    try:
                        plan = backend.plan_scene(user_input, selected_style_prompt, frame_count)
                        st.session_state.plan = plan
                        st.session_state.generated_images = {}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # --- RESULTS ---
    if st.session_state.plan:
        plan = st.session_state.plan
        
        st.divider()
        st.subheader("🔐 Consistency Anchors")
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Subject:**\n{plan.subject_anchor}")
        c2.info(f"**Environment:**\n{plan.environment_anchor}")
        c3.info(f"**Lighting:**\n{plan.lighting_anchor}")

        st.subheader("🎞️ Shot List")
        
        def get_full_prompt(kf):
            if "full_prompt" in kf and len(kf["full_prompt"]) > 20:
                return kf["full_prompt"]
            return f"{selected_style_prompt}. SCENE: {plan.environment_anchor}. SUBJECT: {plan.subject_anchor}. ACTION: {kf['action']}. CAMERA: {kf['camera']}."

        for kf in plan.keyframes:
            full_p = get_full_prompt(kf)
            fid = kf.get('frame_id', '?')
            
            with st.expander(f"Frame {fid}: {kf.get('action', 'Action')}", expanded=True):
                col_text, col_img = st.columns([2, 1])
                
                with col_text:
                    st.code(full_p, language="text")
                    st.caption(f"Camera: {kf.get('camera', 'Static')}")
                
                with col_img:
                    if fid in st.session_state.generated_images:
                        st.image(f"data:image/jpeg;base64,{st.session_state.generated_images[fid]}")
                        # st.image(st.session_state.generated_images[fid], use_column_width=True)
                    else:
                        if st.button(f"🎨 Render {fid}", key=f"btn_{fid}"):
                            if backend:
                                with st.spinner("Rendering..."):
                                    try:
                                        # img_path = backend.render_frame(full_p)
                                        img_b64 = backend.render_frame_to_base64(full_p)
                                        st.session_state.generated_images[fid] = img_b64
                                        # st.session_state.generated_images[fid] = img_path
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed: {e}")

        st.divider()
        if st.button("🚀 Render All Frames (Batch)", type="secondary"):
            if backend:
                progress = st.progress(0)
                for i, kf in enumerate(plan.keyframes):
                    fid = kf.get('frame_id')
                    if fid not in st.session_state.generated_images:
                        try:
                            img_b64 = backend.render_frame_to_base64(get_full_prompt(kf))
                            st.session_state.generated_images[fid] = img_b64
                        except Exception as e:
                            st.warning(f"Frame {fid} failed: {e}")
                    progress.progress((i + 1) / len(plan.keyframes))
                st.success("Done!")
                st.rerun()

if __name__ == "__main__":
    main()