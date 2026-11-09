import os
import io
import base64
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="EraCam AI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAW_FAL_KEY = os.environ.get("FAL_KEY", "fal_sk_2320b9ce87bb449ba31f8986aa139bde:d42ebb50f3e3c234223fd0bffb99a383")
FAL_KEY = RAW_FAL_KEY.strip()
if FAL_KEY.startswith("Fal_sk_"):
    FAL_KEY = "fal_sk_" + FAL_KEY[7:]

FAL_MODEL_ENDPOINT = "https://fal.run/fal-ai/flux-pulid"

ERA_PROMPTS = {
    "1950s": {
        "title": "1950s Mid-Century Kodachrome",
        "prompt": "A photorealistic authentic 1950s Mid-Century Kodachrome color photograph. The person wears elegant 1950s vintage attire with combed vintage styled hair. Setting is a sunny 1950s town street with classic pastel vintage cars parked along the street. Saturated Kodachrome slide film aesthetic with rich, warm analog colors and subtle grain."
    },
    "1920s": {
        "title": "1920s Great Gatsby",
        "prompt": "A photorealistic authentic 1920s vintage photograph. The person wears elegant 1920s vintage attire on a cobblestone avenue with vintage 1920s cars in the background. High contrast silver-gelatin black and white film photograph with authentic grain and tonal depth."
    },
    "1870s": {
        "title": "1870s Victorian Tintype",
        "prompt": "An authentic 1870s Victorian wet-plate tintype portrait photograph. The person wears formal 1870s Victorian attire with high collar. Setting is an antique 19th-century photography studio with heavy draped velvet curtains. Deep sepia wet-plate collodion tones, dark iron shadow depth, and vintage edge vignette."
    },
    "1970s": {
        "title": "1970s Retro Instant Polaroid",
        "prompt": "An authentic 1970s retro instant Polaroid color photograph. The person wears 1970s retro attire with 1970s styled hair. Setting is a warm 1970s wood-paneled room with amber lamp lighting and houseplants. Authentic Polaroid SX-70 look with creamy lifted shadows and warm organic color shift."
    },
    "1980s": {
        "title": "1980s Neon Rock",
        "prompt": "An authentic 1980s 35mm candid color snapshot photograph. The person wears vibrant 1980s retro attire with 1980s voluminous hair. Setting is a vibrant neon-lit street avenue with retro arcade signs at dusk. Direct on-camera xenon snapshot flash, punchy saturated primary colors, and authentic 1980s 35mm film grain."
    },
    "1990s": {
        "title": "1990s Kodak Gold Snapshot",
        "prompt": "An authentic 1990s candid 35mm snapshot photograph. The person wears casual 1990s everyday clothing. Setting is an outdoor 1990s neighborhood park. Authentic Kodak Gold 35mm film tones, warm natural sunlight, subtle grain, and an authentic glowing orange LED corner date stamp reading '94 11 08."
    }
}

class GenerateRequest(BaseModel):
    image_base64: str
    era: str = "1950s"

@app.get("/")
async def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "EraCam Studio Backend Active. index.html not found."}

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "service": "EraCam AI Backend",
        "fal_connected": bool(FAL_KEY)
    }

@app.post("/api/generate")
async def generate_era_portrait(req: GenerateRequest):
    if not req.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    era_info = ERA_PROMPTS.get(req.era, ERA_PROMPTS["1950s"])

    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        
        img_bytes = base64.b64decode(raw_b64)
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        max_dim = 768
        w, h = im.size
        if w > max_dim or h > max_dim:
            if w > h:
                h = int(h * max_dim / w)
                w = max_dim
            else:
                w = int(w * max_dim / h)
                h = max_dim
            im = im.resize((w, h), Image.Resampling.LANCZOS)
            
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        clean_data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    headers = {
        "Authorization": f"Key {FAL_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": era_info["prompt"],
        "reference_image": clean_data_uri,
        "id_weight": 1.0,
        "guidance_scale": 4.0
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(FAL_MODEL_ENDPOINT, headers=headers, json=payload)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"fal.ai API error: {response.text}")
            
            data = response.json()
            image_url = ""
            if "images" in data and len(data["images"]) > 0:
                image_url = data["images"][0].get("url", "")
            elif "image" in data:
                image_url = data["image"].get("url", "")
                
            if not image_url:
                raise HTTPException(status_code=500, detail="fal.ai did not return an image URL")

            return {
                "success": True,
                "image_url": image_url,
                "era": req.era,
                "title": era_info["title"]
            }
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="AI generation timed out on cloud GPU. Please retry.")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
