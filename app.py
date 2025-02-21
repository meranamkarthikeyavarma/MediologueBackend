from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import os
import whisper
from flask_cors import CORS
import io
import tempfile

app = Flask(__name__)
CORS(app)

# Get Hugging Face token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is missing!")

# Hugging Face Model Repository
repo_id = "microsoft/Phi-3-mini-4k-instruct"

# Initialize LLM Client
llm_client = InferenceClient(model=repo_id, token=HF_TOKEN, timeout=120)

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    # Read audio file into memory
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name  
    # Load Whisper model
    model = whisper.load_model("medium")

    # Transcribe audio
    result = model.transcribe(temp_audio_path, language="en")
    transcribed_text = " ".join(segment['text'] for segment in result.get('segments', []))

    os.remove(temp_audio_path)

    # Prompt for structured data extraction
    prompt = f"""You are a medical assistant extracting structured patient data from doctor-patient conversations.
    Extract key details like symptoms, duration, and medication history.

    ### Example 1:
    *Conversation:*
    "I have been feeling very fatigued for the past few days."
    "I feel tired even after a full night's sleep, and I’ve noticed some shortness of breath when climbing stairs."
    "Is the fatigue constant or does it fluctuate?"
    "It’s pretty constant."
    "Sometimes I feel slightly better in the morning, but by the afternoon, I’m exhausted again."
    "Have you had any chest pain or discomfort?"
    "Yes"
    "I’ve had a tight feeling in my chest, but it’s not sharp. It just feels heavy, especially after I exert myself."

    *Extracted Data:*
    - Symptoms: Fatigue, shortness of breath, chest tightness
    - Duration: Fatigue for several days
    - Medication History: None

    ### Example 2:
    *Conversation:*
    "I’ve been having a sore throat and a cough for the past three days."
    "The sore throat is especially worse in the mornings, and it feels scratchy."
    "Does the sore throat get worse with swallowing?"
    "Yes"
    "It does. When I swallow, it feels like something is stuck in my throat. It’s also painful when I talk for too long."
    "Have you had any fever or chills?"
    "Yes, I had a mild fever yesterday, but it went down after I took some paracetamol."

    *Extracted Data:*
    - Symptoms: Sore throat, cough, mild fever
    - Duration: Sore throat and cough for 3 days
    - Medication History: Paracetamol for fever

    ### Now extract data from this conversation:
    *Conversation:*
    {transcribed_text}

    *Extracted Data:*"""

    # Generate structured response using Phi-3
    phi3_response = llm_client.text_generation(prompt, max_new_tokens=200)
    cleaned_response = phi3_response.split("##")[0].strip()

    # Convert extracted data to a dictionary
    structured_dict = {}
    for line in cleaned_response.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            structured_dict[key.strip()] = value.strip()

    return jsonify({
        "message": "Transcription successful",
        "structured_data": structured_dict
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
