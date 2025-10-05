# app.py
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter
from openai import OpenAI
from fastembed import SparseTextEmbedding
from decouple import config

QDRANT_URL = config("QDRANT_URL")
QDRANT_API_KEY = config("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = config("QDRANT_COLLECTION_NAME")
OPENAI_BASE_URL = config("OPENAI_BASE_URL")
OPENAAI_API_KEY = config("OPENAAI_API_KEY")

# Init
router = APIRouter()

bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY)   # adjust if remote

embed_model = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAAI_API_KEY 
)


# serve static files (frontend)
router.mount("/static", StaticFiles(directory="static"), name="static")

@router.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@router.get("/search")
async def search(q: str,
                 request_type: str = None,
                 attachments: str = None,
                 is_internal: str = None,
                 date_from: str = None,
                 date_to: str = None):
    print(date_from)
    query_emb = embed_model.embeddings.create(
        model="openai/E5",
        input=q
    ).data[0].embedding
    sparse_vectors = next(bm25_embedding_model.query_embed(q))
    must_conditions = []

    # attachments filter
    if attachments == "true":
        must_conditions.append(
            models.FieldCondition(
                key="attachments",
                match=models.MatchValue(value=True)
            )
        )

    # attachments filter
    if is_internal == "true":
        must_conditions.append(
            models.FieldCondition(
                key="is_internal",
                match=models.MatchValue(value=True)
            )
        )
    # ticket type filter
    if request_type and request_type != "all":
        must_conditions.append(
            models.FieldCondition(
                key="request_type",
                match=models.MatchValue(value=request_type)
            )
        )

    # date range filter (assuming payload has "created_at" as ISO string or timestamp)
    if date_from or date_to:
        range_filter = {}
        if date_from:
            range_filter["gte"] = date_from
        if date_to:
            range_filter["lte"] = date_to
        must_conditions.append(
            models.FieldCondition(
                key="date",
                range=models.DatetimeRange(**range_filter)
            )
        )
    print(must_conditions)
    hits = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
            query=models.SparseVector(**sparse_vectors.as_object()),
            using="multi_bm25",
            limit=100,
            ),
            models.Prefetch(
            query=models.SparseVector(**sparse_vectors.as_object()),
            using="bm25",
            limit=50,
            ),
            models.Prefetch(
            query=query_emb,
            using="multi_e5",
            limit=50,
            ),
        ],
        query=query_emb,
        using="e5",
        query_filter=Filter(must=must_conditions) if must_conditions else None,
        limit=9
    )
    results = [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload
        }
        for hit in hits.points
    ]
    return JSONResponse(content=results)
