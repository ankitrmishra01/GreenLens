import google.generativeai as genai
from config import get_settings
from typing import Optional, Dict
import json
import base64
import io
from PIL import Image
import imagehash

def get_image_hash(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    return str(imagehash.phash(image))

settings = get_settings()

def initialize_gemini():
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)

def scan_receipt_or_food(image_bytes: bytes, mime_type: str) -> Optional[Dict]:
    """
    Uses Gemini 1.5 Flash to analyze an image of a receipt, food, or transit ticket,
    and returns a structured JSON estimating its carbon footprint.
    """
    if not settings.GEMINI_API_KEY:
        return None
        
    try:
        initialize_gemini()
        model = genai.GenerativeModel("gemini-flash-latest")
        
        prompt = """
        You are an eco-friendly AI assistant. The user has uploaded an image of a grocery receipt, a meal, or a transit ticket.
        Please analyze it and estimate the carbon footprint.
        
        Respond ONLY with a valid JSON object in this exact format:
        {
          "activity_type": "food", // Must be one of: transport, food, electricity, purchases, waste
          "value": 1.5, // Estimated numerical value for the unit
          "unit": "meal", // e.g., km, meal, kWh, items, kg
          "description": "Vegetarian meal with rice and curries", // Short description of what you found
          "confidence": 0.85, // Your confidence in this estimate (0.0 to 1.0)
          "sdg_goal": "SDG 12: Responsible Consumption and Production", // The UN Sustainable Development Goal this aligns with
          "receipt_id": "12345-ABC" // Extracted receipt or transaction ID if it is a receipt. Otherwise, return null.
        }
        """
        
        response = model.generate_content([
            prompt,
            {"mime_type": mime_type, "data": image_bytes}
        ])
        
        # Clean up the response in case it contains markdown formatting like ```json
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        result = json.loads(text.strip())
        result["image_hash"] = get_image_hash(image_bytes)
        return result
        
    except Exception as e:
        print(f"Vision API Error: {e}")
        return None
