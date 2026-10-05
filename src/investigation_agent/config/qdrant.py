"""Connection to Qdrant (local Docker, port 6333)."""

from qdrant_client import QdrantClient, models

from investigation_agent.config.settings import Settings


def ensure_collection(client: QdrantClient, settings: Settings) -> None:
    """Create the configured hybrid collection (named dense + BM25 sparse vectors) if it does
    not exist. If it exists without the dense vector or with another size, fail: mixing
    vectors from two embedding models silently breaks search."""
    name, dim = settings.qdrant_collection, settings.h200_embed_dim
    dense, sparse = settings.qdrant_dense_vector, settings.qdrant_sparse_vector
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config={dense: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={sparse: models.SparseVectorParams(modifier=models.Modifier.IDF)})
        return
    params = client.get_collection(name).config.params
    vectors = params.vectors if isinstance(params.vectors, dict) else {}
    if dense not in vectors or vectors[dense].size != dim:
        raise ValueError(f"Collection {name!r} has no dense vector {dense!r} of size {dim}: "
                         "was it built for another schema or embedding model?")
    if sparse not in (params.sparse_vectors or {}):
        raise ValueError(f"Collection {name!r} has no sparse vector {sparse!r}")


def get_qdrant_client(settings: Settings) -> QdrantClient:
    """Build a Qdrant client from the given settings, with the configured collection ready."""
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout)
    ensure_collection(client, settings)
    return client
