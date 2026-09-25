from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "keras_model.h5"
LABELS_PATH = APP_DIR / "labels.txt"

st.set_page_config(page_title="CampusFit AI", page_icon="🧥", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#18251f; --muted:#64736b; --green:#18734f; --cream:#f6f7f2; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp { background:var(--cream); }
.block-container { max-width:1120px; padding-top:2rem; }
.hero { padding:2rem 2.2rem; border-radius:24px; background:linear-gradient(120deg,#123f31,#207656); color:white; margin-bottom:1.5rem; }
.hero h1 { font-family:Manrope,sans-serif; color:white; font-size:2.6rem; margin:.45rem 0; }
.hero p { color:#d4e9de; max-width:650px; font-size:1.05rem; }
.eyebrow { letter-spacing:.14em; text-transform:uppercase; font-weight:700; font-size:.72rem; color:#9fe0bc; }
.panel { background:white; border:1px solid #e8ece5; padding:1.25rem 1.4rem; border-radius:18px; margin:.4rem 0 1rem; }
div[data-testid="stMetric"] { background:white; border:1px solid #e8ece5; padding:1rem; border-radius:16px; }
.small-note { color:#64736b; font-size:.88rem; }
footer {visibility:hidden;}
</style>
<div class="hero"><div class="eyebrow">AI • Style • Student life</div><h1>CampusFit AI</h1><p>A practical outfit decision assistant. Scan a clothing item, set the context for your day, and get a recommendation with the model’s confidence and reasoning.</p></div>
""", unsafe_allow_html=True)

class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, groups=1, **kwargs):
        # Ignore Conv2D-only groups metadata in this exported depthwise layer.
        super().__init__(*args, **kwargs)


@st.cache_resource
def load_model_and_labels(model_path: str, labels_path: str):
    from teachable_machine import TeachableMachine
    machine = TeachableMachine(model_path=model_path, labels_file_path=labels_path)
    model = machine._model
    labels = [line.strip() for line in Path(labels_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError("labels.txt is empty")
    return model, labels

def clean_label(label: str) -> str:
    parts = label.strip().split(maxsplit=1)
    return parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else label.strip()

def category_for(label: str) -> str:
    text = label.casefold()
    formal = ("formal", "business", "blazer", "suit", "office", "professional", "shirt")
    casual = ("casual", "t-shirt", "tshirt", "tee", "jeans", "hoodie", "sneaker", "sportswear", "streetwear")
    if any(word in text for word in formal):
        return "Polished"
    if any(word in text for word in casual):
        return "Casual"
    return "Versatile"

def make_recommendation(label: str, occasion: str, temp: int, comfort: str):
    category = category_for(label)
    reasons = [f"The model recognized **{label}** as a {category.lower()} option."]
    if occasion == "Regular lectures" and category == "Casual" and temp > 30:
        advice = "Casual wear is a practical choice for a lecture. Pair it with comfortable shoes."
        reasons.append("Casual wear keeps the outfit relaxed and practical for lectures.")
    elif occasion == "Presentation" and category == "Casual":
        advice = "Add a structured layer, such as a blazer or neat overshirt, to make the outfit feel presentation-ready."
        reasons.append("A presentation usually benefits from a more polished look.")
    elif occasion == "Presentation":
        advice = "This looks suitable for a presentation. Keep the rest of the outfit neat and comfortable."
        reasons.append("Its polished style fits your presentation.")
    elif occasion == "Exam / test":
        advice = "Prioritize easy movement and comfortable shoes so you can focus through the exam."
        reasons.append("Comfort matters during a long exam day.")
    elif occasion == "Campus hangout":
        advice = "This is a relaxed choice for a social campus day. Pair it with shoes you can walk in."
        reasons.append("Its relaxed style works well for a hangout.")
    else:
        advice = "This is a practical choice for lectures. Pair it with comfortable shoes for moving around campus."
        reasons.append("It works for a regular class day.")
    if temp < 20:
        advice += " Bring a warm outer layer for the cool weather."
        reasons.append(f"At {temp}°C, an extra layer should help you stay warm.")
    elif temp >= 30:
        advice += " Choose breathable fabric and carry water in the heat."
        reasons.append(f"At {temp}°C, breathable layers should feel more comfortable.")
    else:
        reasons.append(f"At {temp}°C, light layering should be comfortable.")
    if comfort == "Maximum comfort":
        advice += " Choose your softest fabrics and supportive footwear."
        reasons.append("You selected maximum comfort as a priority.")
    return advice, reasons

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("### 01 · Your day")
    occasion = st.selectbox("What’s on your schedule?", ["Regular lectures", "Exam / test", "Presentation", "Campus hangout"])
    temperature = st.slider("Temperature outside", 10, 40, 25, format="%d °C")
    comfort = st.radio("Style priority", ["Balanced", "Maximum comfort"], horizontal=True)
    st.markdown('<p class="small-note">Recommendations are guidance; follow your campus dress code and personal preferences.</p>', unsafe_allow_html=True)
with right:
    st.markdown("### 02 · Scan a clothing item")
    uploaded = st.file_uploader("Choose a clear photo", type=["jpg", "jpeg", "png"], help="The image is processed in this app session and is not saved by this app.")
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Selected clothing photo", use_container_width=True)
        except (UnidentifiedImageError, OSError) as exc:
            st.error(f"This image could not be opened: {exc}")
            image = None
    else:
        image = None
        st.markdown('<div class="panel"><b>Tip</b><br><span class="small-note">Use a well-lit image with the clothing item clearly visible. The model works best on the kinds of images it was trained with.</span></div>', unsafe_allow_html=True)

if uploaded and image is not None:
    st.markdown("### 03 · AI analysis")
    try:
        if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
            raise FileNotFoundError("Place keras_model.h5 and labels.txt beside app.py.")
        model, class_names = load_model_and_labels(str(MODEL_PATH), str(LABELS_PATH))
        prepared = ImageOps.fit(image, (224, 224), method=Image.Resampling.LANCZOS)
        batch = np.expand_dims((np.asarray(prepared, dtype=np.float32) / 127.5) - 1.0, axis=0)
        scores = np.asarray(model.predict(batch, verbose=0))[0].reshape(-1)
        if len(scores) != len(class_names):
            raise ValueError(f"Model returned {len(scores)} scores but labels.txt contains {len(class_names)} labels.")
        order = np.argsort(scores)[::-1]
        top = [(clean_label(class_names[i]), float(scores[i])) for i in order[:min(3, len(order))]]
        label, confidence = top[0]
        a, b = st.columns([1, 2])
        with a:
            st.metric("Top match", label)
            st.metric("Model confidence", f"{confidence:.0%}")
        with b:
            st.markdown('<div class="panel"><b>What the model considered</b>', unsafe_allow_html=True)
            for name, score in top:
                st.write(f"**{name}** · {score:.1%}")
                st.progress(min(max(score, 0.0), 1.0))
            st.markdown('</div>', unsafe_allow_html=True)
        if confidence < 0.55:
            st.warning("The model is unsure about this image. Try a clearer photo with the item centered; treat this result as a rough suggestion.")
        advice, reasons = make_recommendation(label, occasion, temperature, comfort)
        st.markdown("### Your outfit plan")
        st.success(advice)
        with st.expander("Why this recommendation?"):
            for reason in reasons:
                st.write(f"• {reason}")
        st.caption("AI results can be inaccurate, especially for classes or photos unlike the training examples.")
    except Exception as exc:
        st.error(f"Could not analyze this photo: {type(exc).__name__}: {exc}")

with st.expander("About this project"):
    st.write("CampusFit AI demonstrates image classification with a Teachable Machine model and rule-based decision support. The classifier identifies a clothing class; transparent rules combine it with schedule, temperature, and comfort preference to produce a practical suggestion.")
