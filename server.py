from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import uuid
import os
import traceback

app = Flask(__name__)
CORS(app)

# Allow large files (Railway-safe)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Whisper model...")

# ✅ BALANCED MODEL (GOOD ACCURACY + SPEED)
model = WhisperModel(
    "base",          # better than tiny
    device="cpu",
    compute_type="int8"
)

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.route("/")
def home():
    return "Speech-to-Text Server Running OK"

# -----------------------------
# TRANSCRIBE + TRANSLATE
# -----------------------------
@app.route("/transcribe", methods=["POST"])
def transcribe():

    filepath = None

    try:
        # -----------------------------
        # CHECK FILE
        # -----------------------------
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        if not file or file.filename.strip() == "":
            return jsonify({
                "success": False,
                "error": "Empty file"
            }), 400

        # -----------------------------
        # GET TARGET LANGUAGE FROM ANDROID
        # -----------------------------
        target_lang = request.form.get("target_lang", "id")

        # -----------------------------
        # SAVE FILE
        # -----------------------------
        ext = os.path.splitext(file.filename)[1]
        if ext == "":
            ext = ".mp3"

        filename = f"{uuid.uuid4()}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        print("Saved:", filepath)

        # -----------------------------
        # WHISPER TRANSCRIPTION
        # -----------------------------
        segments, info = model.transcribe(
            filepath,
            task="transcribe",
            language=None,  # AUTO DETECT LANGUAGE
            beam_size=5,
            best_of=5,
            vad_filter=True,
            condition_on_previous_text=False
        )

        text = " ".join(segment.text for segment in segments).strip()

        print("Original text:", text)
        print("Detected language:", info.language)

        # -----------------------------
        # TRANSLATION
        # -----------------------------
        translated = text

        if text.strip():
            try:
                translated = GoogleTranslator(
                    source="auto",
                    target=target_lang
                ).translate(text)
            except Exception as e:
                print("Translation error:", e)

        # -----------------------------
        # RESPONSE
        # -----------------------------
        return jsonify({
            "success": True,
            "original": text,
            "translated": translated,
            "language": info.language,
            "target_language": target_lang
        })

    except Exception as e:
        print("ERROR:")
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        # -----------------------------
        # CLEANUP FILE
        # -----------------------------
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                print("Temp file deleted")
        except Exception as e:
            print("Delete error:", e)

# -----------------------------
# START SERVER (RAILWAY)
# -----------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )