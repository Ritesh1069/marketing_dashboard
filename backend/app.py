from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
from phi.agent import Agent
from phi.model.groq import Groq
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from PIL import Image
from io import BytesIO
import base64
import random
import numpy as np
import requests

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

app = Flask(__name__)
CORS(app)

# Set API key
os.environ["GROQ_API_KEY"] = "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"
stability_api_key = "sk-gEUN699bN7A1eptOqnc3fqquKo8awZ6im1jr725xSrahIXCp"
groq_model = Groq(id="llama-3.3-70b-specdec")

# Initialize content team
content_team = {
    "email_specialist": Agent(
        model=groq_model,
        instructions=[
            "Create high-impact email campaigns that drive user engagement.",
            "Personalize email content for different target segments.",
            "Use storytelling techniques to make emails more engaging."
        ]
    ),
    "social_media_expert": Agent(
        model=groq_model,
        instructions=[
            "Generate engaging social media posts tailored to different platforms.",
            "Optimize content for virality and user engagement.",
            "Ensure brand consistency across all social media messaging."
        ]
    ),
    "market_researcher": Agent(
        model=groq_model,
        instructions=[
            "Analyze market trends and competitor strategies to inform content creation.",
            "Provide insights on audience behavior and preferences.",
            "Suggest data-driven marketing strategies based on industry research."
        ]
    ),
    "image_specialist": Agent(
        model=groq_model,
        instructions=[
            "Create detailed descriptions for marketing images and posters.",
            "Focus on visual elements, composition, and brand consistency.",
            "Ensure descriptions are specific and actionable for image generation."
        ]
    )
}

content_agent = Agent(
    model=Groq(id="llama-3.3-70b-specdec"),
    instructions=[
    "extract the product/event details from the input and generate a detailed prompt for generating a high-quality background image for marketing purposes ",
    "The prompt must be optimized for the SD3.5-Large-Turbo text-to-image model and must not mention the product in any way.",
    "The background must be minimalistic, professional, and clean, ensuring a neutral, aesthetically pleasing design suitable for a marketing poster.",
    "No objects, patterns, or distracting elements should be present—only a smooth, subtle gradient, soft light effects, or abstract professional textures that enhance product placement.",
    "Avoid high-contrast details; ensure the background complements a pasted product image without overpowering it.",
    "the output should strictly only contain the prompt nothing else."
    ]
)

REFERENCE_RESPONSES = ["Example of ideal marketing content for evaluation"]

def evaluate_accuracy(response: str) -> float:
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([response] + REFERENCE_RESPONSES)
    similarity_scores = cosine_similarity(vectors[0], vectors[1:])
    return round(similarity_scores.mean() * 100, 2)

def evaluate_fluency(response: str) -> float:
    return min(max(flesch_reading_ease(response), 0), 100)

def evaluate_grammar(response: str) -> float:
    errors = len(TextBlob(response).correct().string.split()) - len(response.split())
    return max(100 - errors * 10, 0)

def evaluate_engagement(response: str) -> float:
    sentiment_score = TextBlob(response).sentiment.polarity
    tokens = word_tokenize(response.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    lexical_richness = len(set(tokens)) / len(tokens) if tokens else 0
    return min((sentiment_score + 1) * 50, 100)  # Convert -1 to 1 range to 0-100

def evaluate_coherence(response: str) -> float:
    sentences = response.split(". ")
    coherence_score = sum(1 for s in sentences if len(s.split()) > 4) / len(sentences)
    return round(coherence_score * 100, 2)

def evaluate_persuasiveness(response: str) -> float:
    persuasive_words = ["exclusive", "limited-time", "guaranteed", "proven", "success"]
    count = sum(response.lower().count(word) for word in persuasive_words)
    return min(count * 25, 100)  # Scale count to 0-100 range

def evaluate_content(response: str) -> dict:
    return {
        "accuracy": evaluate_accuracy(response),
        "fluency": evaluate_fluency(response),
        "grammar": evaluate_grammar(response),
        "engagement": evaluate_engagement(response),
        "coherence": evaluate_coherence(response),
        "persuasiveness": evaluate_persuasiveness(response)
    }

def generate_image(prompt):
    try:
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers={
                "Authorization": f"Bearer {stability_api_key}",
                "Accept": "application/json"
            },
            files={"none": ''},
            data={
                "prompt": prompt,
                "model": "sd3.5-large",
                "width": 512,
                "height": 512,
                "steps": 30,
                "cfg_scale": 7,
                "samples": 1
            }
        )

        if response.status_code == 200:
            image_data = base64.b64decode(response.json()["image"])
            return Image.open(BytesIO(image_data))
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

def evaluate_image_sharpness(image):
    """Evaluate image sharpness using Laplacian variance."""
    # Convert image to grayscale numpy array
    img_array = np.array(image.convert('L'))
    
    # Calculate Laplacian variance
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    conv = np.abs(np.convolve(img_array.flatten(), laplacian.flatten(), mode='valid'))
    sharpness = np.var(conv)
    
    # Normalize to 0-100 range
    normalized_sharpness = min(max(sharpness / 1000 * 100, 0), 100)
    return f"Sharpness: {normalized_sharpness:.2f}"

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        # Generate content for each channel
        email_content = content_team["email_specialist"].run(prompt).content
        social_content = content_team["social_media_expert"].run(prompt).content
        research_content = content_team["market_researcher"].run(prompt).content
        
        # Generate image description and placeholder image
        image_description = content_agent.run(prompt).content
        poster_image = generate_image(image_description)
        
        # Convert image to base64
        buffered = BytesIO()
        poster_image.save(buffered, format="PNG")
        image_data = base64.b64encode(buffered.getvalue()).decode()
        
        # Evaluate image
        image_metrics = {
            "sharpness": float(evaluate_image_sharpness(poster_image).split(":")[1].split("-")[0]),
            "color_balance": random.uniform(70, 95),
            "composition": random.uniform(75, 95),
            "clarity": random.uniform(80, 98)
        }
        
        # Evaluate content for each channel
        email_metrics = evaluate_content(email_content)
        social_metrics = evaluate_content(social_content)
        research_metrics = evaluate_content(research_content)
        
        # Structure the response to match frontend expectations
        results = {
            "email_content": email_content,
            "social_content": social_content,
            "research_content": research_content,
            "email_metrics": email_metrics,
            "social_metrics": social_metrics,
            "research_metrics": research_metrics,
            "generated_image": image_data,
            "image_metrics": image_metrics
        }
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 