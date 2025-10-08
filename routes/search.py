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
    api_key=QDRANT_API_KEY,
    timeout=60)   # adjust if remote

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
                 ticket_id: str = None,
                 date_from: str = None,
                 date_to: str = None,
                 true_tickets: str = None):
    must_conditions = []
    if ticket_id:
        print(ticket_id)
        must_conditions.append(
            models.FieldCondition(
                key="ticket_id",
                match=models.MatchValue(value=int(ticket_id))
            )
        )
        hits = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query_filter=Filter(must=must_conditions) if must_conditions else None,
        limit=10
        )
        for point in hits.points:
            point.score = 1.000
    else:
        query_emb = embed_model.embeddings.create(
            model="openai/E5",
            input=q
        ).data[0].embedding
        sparse_vectors = next(bm25_embedding_model.query_embed(q))
        

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
        # if int(ticket_id) :
        #     must_conditions.append(
        #         models.FieldCondition(
        #             key="ticket_id",
        #             match=models.MatchValue(value=ticket_id)
        #         )
        #     )
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
                # models.Prefetch(
                # query=query_emb,
                # using="multi_e5",
                # limit=50,
                # ),
                # models.Prefetch(
                # query=query_emb,
                # using="e5",
                # limit=50,
                # ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            # query=query_emb,
            # using="e5",
            query_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=50
        )
    results = []
    ticket_ids = set()
    predicted_ids = []
    predicted_scores = []
    for point in hits.points:
        if point.payload['ticket_id'] not in ticket_ids:
            predicted_ids.append(point.payload['ticket_id'])
            predicted_scores.append(point.score)
            ticket_ids.add(point.payload['ticket_id'])
            results.append({
            "id": point.id,
            "score": point.score,
            "payload": point.payload
        })
    import numpy as np
    import re
    precision_at_10 = None
    recall_at_10 = None
    if true_tickets:
        y_true_ids = [int(t) for t in re.findall(r"\d+", true_tickets)]
        print(y_true_ids)
        predicted_ids = np.array(predicted_ids)
        predicted_scores = np.array(predicted_scores)
        ranked_indices = np.argsort(predicted_scores)[::-1]
        top_k = 10
        top_k_indices = ranked_indices[:top_k]

        # Step 2: Get top-k predicted IDs
        top_k_predicted_ids = predicted_ids[top_k_indices]

        # Step 3: Mark as relevant if in ground truth
        y_true_top_k = np.array([1 if tid in y_true_ids else 0 for tid in top_k_predicted_ids])

        # Step 4: Compute precision@10 and recall@10
        precision_at_10 = np.sum(y_true_top_k) / top_k
        recall_at_10 = np.sum(y_true_top_k) / len(y_true_ids)

        print(f"Top-10 predicted IDs: {top_k_predicted_ids}")
        print(f"Precision@10: {precision_at_10:.3f}")
        print(f"Recall@10: {recall_at_10:.3f}")

    return JSONResponse(content={"results" : results[:9], "metrics": {
        "precision_at_10": f"{precision_at_10:.3f}",
        "Recall_at_10": f"{recall_at_10:.3f}"
    }})
