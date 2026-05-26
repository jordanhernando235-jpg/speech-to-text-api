from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import os
import uuid
import traceback

app = Flask(__name__)
CORS(app)

# Allow big audio uploads (2–10 minutes safe)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Whisper model...")

# 🔥 BEST BALANCE FOR RAILWAY
model = WhisperModel(
    "base",              # better than tiny for accuracy
    device="cpu",
    compute_type="int8"
)

@app.route("/")
def home():
    return "Speech-to-Text API Running"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    filepath = None

    try:
        # ---------------------------
        # CHECK FILE
        # ---------------------------
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        if file.filename.strip() == "":
            return jsonify({
                "success": False,
                "error": "Empty filename"
            }), 400

        # ---------------------------
        # SAVE FILE
        # ---------------------------
        ext = os.path.splitext(file.filename)[1]
        if ext == "":
            ext = ".mp3"

        filename = f"{uuid.uuid4()}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        print("Saved:", filepath)

        # ---------------------------
        # TRANSCRIBE (AUTO LANGUAGE DETECT)
        # ---------------------------
        segments, info = model.transcribe(
            filepath,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False
        )

        text = " ".join([seg.text for seg in segments]).strip()

        print("Detected language:", info.language)
        print("Text:", text)

        # ---------------------------
        # TRANSLATION (OPTIONAL)
        # ---------------------------
        target_lang = request.form.get("target_lang", "id")

        translated = text

        if text.strip():
            try:
                translated = GoogleTranslator(
                    source="auto",
                    target=target_lang
                ).translate(text)
            except Exception as e:
                print("Translation error:", e)
                translated = text

        # ---------------------------
        # RESPONSE FORMAT (IMPORTANT FOR ANDROID)
        # ---------------------------
        return jsonify({
            "success": True,
            "original": text,
            "translated": translated,
            "detected_language": info.language,
            "target_language": target_lang
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        # cleanup file
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print("Delete error:", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )