from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import uuid
import os
import traceback

app = Flask(__name__)
CORS(app)

# Optional: allow larger uploads
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Whisper model...")

# tiny = fastest/lightest
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

    try:
        # Check file exists
        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        # Check filename
        if file.filename == "":
            return jsonify({
                "error": "Empty filename"
            }), 400

        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save uploaded audio
        file.save(filepath)

        print(f"Saved file: {filepath}")

        # Speech-to-text
	segments, info = model.transcribe(
    	filepath,
    	beam_size=1,
    	vad_filter=True
	)
      
        text = ""

        for segment in segments:
            text += segment.text + " "

        text = text.strip()

        print("Transcribed text:", text)

        # Translate
        translated = GoogleTranslator(
            source="auto",
            target="id"
        ).translate(text)

        print("Translated text:", translated)

        # Return response
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
        # Delete temp file
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
                print("Temporary file deleted.")
        except:
            pass

if __name__ == "__main__":

    # Railway dynamic port support
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )