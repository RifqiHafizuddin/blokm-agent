"""Step 7: build the vectors (capstone Module 2 pattern).

places_text   MiniLM (384-d) over name, category, cuisine, vibe and summary
places_photo  CLIP ViT-B/32 (512-d) over single-place video thumbnails
Vectors are L2-normalised, so Chroma's default L2 ranking equals cosine ranking.
They are saved as plain JSON (data/vectors.json); the app loads them into in-memory Chroma
collections at startup. That keeps the repo free of binary index files (Hugging Face requires
Xet/LFS for binaries) and the index is rebuilt in well under a second."""
import io

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from common import DATA, http, read_json, write_json

VECTORS = DATA / "vectors.json"
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
        "wifi": s["wifi"]["value"], "wfc_score": p["wfc_score"],
        "trending_score": p["buzz"]["trending_score"], "favorite_score": p["buzz"]["favorite_score"],
    }


@torch.no_grad()
def clip_embed(model, processor, images: list) -> np.ndarray:
    # vision encoder -> projection: CLIP's image embedding, stable across transformers versions
    pooled = model.vision_model(pixel_values=processor(images=images, return_tensors="pt")["pixel_values"]).pooler_output
    feats = model.visual_projection(pooled)
    return (feats / feats.norm(dim=-1, keepdim=True)).numpy()


def rounded(vecs: np.ndarray) -> list:
    return np.round(vecs, 5).tolist()  # 5 decimals: identical rankings, ~half the file size


def main():
    places = read_json(DATA / "places.json")

    text_model = SentenceTransformer(TEXT_MODEL)
    docs = [place_text(p) for p in places]
    text = {"ids": [p["id"] for p in places], "documents": docs, "metadatas": [metadata(p) for p in places],
            "embeddings": rounded(text_model.encode(docs, normalize_embeddings=True))}
    print(f"places_text: {len(text['ids'])} vectors")

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
    embs = [clip_embed(clip, processor, images[i:i + 16]) for i in range(0, len(images), 16)]
    photo = {"ids": ids, "metadatas": metas, "embeddings": rounded(np.vstack(embs)) if embs else []}
    print(f"places_photo: {len(ids)} vectors from {len({m['place_id'] for m in metas})} places")
    write_json(VECTORS, {"places_text": text, "places_photo": photo})
    print(f"saved {VECTORS.relative_to(DATA.parent)} ({VECTORS.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
