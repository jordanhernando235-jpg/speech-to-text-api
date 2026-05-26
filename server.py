from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import uuid
import os
import traceback

app = Flask(__name__)
CORS(app)

# Allow large uploads
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Whisper model...")

# Whisper model
# tiny = fastest
# base = better accuracy
model = WhisperModel(
    "base",
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
        # Check uploaded file
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        # Empty filename check
        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "Empty filename"
            }), 400

        # Create unique filename
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save audio file
        file.save(filepath)

        print(f"Saved file: {filepath}")

        # Transcribe audio
        segments, info = model.transcribe(
            filepath,
            beam_size=1,
            vad_filter=True
        )

        # Combine text
        text = ""

        for segment in segments:
            text += segment.text + " "

        text = text.strip()

        print("Transcribed text:", text)

        # Translate text
        translated = GoogleTranslator(
            source="auto",
            target="id"
        ).translate(text)

        print("Translated text:", translated)

        # Return result
        return jsonify({
            "success": True,
            "text": text,
            "translated": translated,
            "language": info.language
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        # Delete uploaded temp file
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                print("Temporary file deleted.")
        except Exception as delete_error:
            print("Delete error:", delete_error)

if __name__ == "__main__":

    # Railway dynamic port
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )