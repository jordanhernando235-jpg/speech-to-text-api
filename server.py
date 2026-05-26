from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import uuid
import os
import traceback

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Whisper model (base - slower but accurate)...")

# 🔥 BETTER ACCURACY FOR 2 MIN AUDIO
model = WhisperModel(
    "base",          # change from tiny → base
    device="cpu",
    compute_type="int8"
)

@app.route("/")
def home():
    return "Speech-to-Text Server Running OK"

@app.route("/transcribe", methods=["POST"])
def transcribe():

    filepath = None

    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]

        if not file or file.filename.strip() == "":
            return jsonify({"success": False, "error": "Empty file"}), 400

        # keep real extension
        ext = os.path.splitext(file.filename)[1]
        if ext == "":
            ext = ".mp3"

        filename = f"{uuid.uuid4()}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        print(f"Saved file: {filepath}")

        # 🔥 SLOWER BUT MORE ACCURATE SETTINGS
        segments, info = model.transcribe(
            filepath,
            beam_size=5,                  # better accuracy
            best_of=5,
            vad_filter=True,
            condition_on_previous_text=True
        )

        text = " ".join(segment.text for segment in segments).strip()

        print("Original text:", text)

        # language from whisper auto detect
        detected_lang = info.language

        # translation target from android
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

        return jsonify({
            "success": True,
            "original": text,
            "translated": translated,
            "detected_language": detected_lang
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
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