from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import uuid
import os
import traceback

app = Flask(__name__)
CORS(app)

# Allow large uploads (recommended max ~50MB for stability)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Whisper model... (tiny for speed)")

# ✅ FAST MODEL FOR 2-MIN AUDIO
model = WhisperModel(
    "tiny",   # FASTEST OPTION
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
        # Check file exists
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

        # Save file
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        print(f"Saved file: {filepath}")

        # 🔥 FASTER TRANSCRIPTION SETTINGS
        segments, info = model.transcribe(
            filepath,
            beam_size=1,
            best_of=1,
            vad_filter=True,
            condition_on_previous_text=False
        )

        # Combine text
        text = " ".join(segment.text for segment in segments).strip()

        print("Transcribed text:", text)

        # Translation (safe)
        translated = text

        try:
            if text.strip():
                translated = GoogleTranslator(
                    source="auto",
                    target="id"
                ).translate(text)
        except Exception as e:
            print("Translation error:", e)
            translated = text

        return jsonify({
            "success": True,
            "text": text,
            "translated": translated,
            "language": info.language
        })

    except Exception as e:
        print("ERROR:")
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
                print("Temporary file deleted.")
        except Exception as e:
            print("Delete error:", e)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False  # IMPORTANT for Railway
    )