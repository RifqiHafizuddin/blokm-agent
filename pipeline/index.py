"""Step 7: build the vector indexes (capstone Module 2 pattern).

places_text   MiniLM (384-d) over name, category, cuisine, vibe and summary
places_photo  CLIP ViT-B/32 (512-d) over single-place video thumbnails
Vectors are L2-normalised, so Chroma's default L2 ranking equals cosine ranking."""
import io
import shutil

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from common import DATA, http, read_json

CHROMA_DIR = DATA / "chroma"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"


def place_text(p: dict) -> str:
    where = p.get("address") or p.get("area_hint") or "Blok M"
    return (f"{p['name']}. {p['category']}; cuisine: {', '.join(p['cuisine'])}. "
            f"Vibe: {', '.join(p['vibe_tags'])}. {p['summary']} Location: {where}.")


def metadata(p: dict) -> dict:
    s = p["signals"]
    return {
        "place_id": p["id"], "name": p["name"], "category": p["category"],
        "lat": p["lat"], "lon": p["lon"], "location_precision": p["location_precision"],
        "price_idr": p["price_idr_per_person"] or 0, "halal": s["halal"]["value"],
        "halal_confidence": s["halal"]["confidence"], "vegetarian": s["vegetarian_friendly"]["value"],
        "wifi": s["wifi"]["value"], "wfc_score": p["wfc_score"], "viral_score": p["viral"]["score"],
    }


@torch.no_grad()
def clip_embed(model, processor, images: list) -> np.ndarray:
    # vision encoder -> projection: CLIP's image embedding, stable across transformers versions
    pooled = model.vision_model(pixel_values=processor(images=images, return_tensors="pt")["pixel_values"]).pooler_output
    feats = model.visual_projection(pooled)
    return (feats / feats.norm(dim=-1, keepdim=True)).numpy()


def main():
    import chromadb

    places = read_json(DATA / "places.json")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)  # rebuild from scratch, no duplicate vectors
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    text_model = SentenceTransformer(TEXT_MODEL)
    vecs = text_model.encode([place_text(p) for p in places], normalize_embeddings=True)
    text_col = client.create_collection("places_text")
    text_col.add(ids=[p["id"] for p in places], embeddings=vecs.tolist(),
                 documents=[place_text(p) for p in places], metadatas=[metadata(p) for p in places])
    print(f"places_text: {text_col.count()} vectors")

    clip = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    ids, images, metas = [], [], []
    for p in places:
        for j, t in enumerate(p.get("thumbnails", [])):
            try:
                img = Image.open(io.BytesIO(http("GET", t["url"], retries=2).content)).convert("RGB")
            except Exception as e:
                print(f"  skip thumbnail for {p['name']}: {str(e)[:60]}")
                continue
            ids.append(f"{p['id']}#{j}")
            images.append(img)
            metas.append({**metadata(p), "thumbnail": t["url"], "video_url": t["video_url"]})
    photo_col = client.create_collection("places_photo")
    for i in range(0, len(images), 16):
        photo_col.add(ids=ids[i:i + 16], embeddings=clip_embed(clip, processor, images[i:i + 16]).tolist(),
                      metadatas=metas[i:i + 16])
    print(f"places_photo: {photo_col.count()} vectors from {len({m['place_id'] for m in metas})} places")


if __name__ == "__main__":
    main()
