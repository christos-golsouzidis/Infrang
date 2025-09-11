from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import uvicorn
from infrang_core import Infrang
import dotenv
import os
import logging

version = '1.2.4'

# Setup
dotenv.load_dotenv()
app = FastAPI(title="Infrang API", version=version)
logger = logging.getLogger(__name__)


# Models
class InfrangConfig(BaseModel):
    dense_model_name: Optional[str] = "BAAI/bge-small-en-v1.5"
    sparse_model_name: Optional[str] = "prithivida/Splade_PP_en_v1"
    paraphrase_model_name: Optional[str] = "ramsrigouthamg/t5_paraphraser"
    generate_model_name: Optional[str] = "llama-3.3-70b-versatile"
    parallel: Optional[int] = 4
    groq_api_key: Optional[str] = None


# Helper function
def get_infrang_instance(collection: str, config: InfrangConfig) -> Infrang:

    groq_api_key = config.groq_api_key or os.getenv("GROQ_API_KEY")
    return Infrang(
        collection=collection,
        dense_model_name=config.dense_model_name,
        sparse_model_name=config.sparse_model_name,
        paraphrase_model_name=config.paraphrase_model_name,
        generate_model_name=config.generate_model_name,
        parallel=config.parallel,
        groq_api_key=groq_api_key
    )


@app.get("/")
async def root():
    return {"message": "Infrang API, version {}".format(version)}


@app.get("/collections")
async def get_collections():
    if not os.path.exists(os.path.join('data','collection')):
        os.makedirs(os.path.join('data','collection'))
    _, collections, _ =  next(os.walk(os.path.join('data', 'collection')))
    try:
        return {
                "message": "List of collections",
                "collections": collections
            }
    except Exception as e:
        logger.error(f"Error on 'get collections': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Get all sources of a database/collection
@app.get("/collections/{collection}")
async def get_sources(
    collection: str,
    config: InfrangConfig = None,
):
    if config is None:
        config = InfrangConfig()
    try:
        infrang = get_infrang_instance(collection, config)
        sources = infrang.get_sources()
        return {
            "message": "Sources of {}".format(collection),
            "collection" : collection,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error on 'get sources': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    
# Create a new database/collection
@app.post("/collections/{collection}/{path:path}")
async def create_collection(
    collection: str,
    path: str,
    config: InfrangConfig = None,
    overwrite: bool = False
):
    
    if config is None:
        config = InfrangConfig()
    try:
        infrang = get_infrang_instance(collection, config)
        infrang.create(kb_path=path, overwrite=overwrite)
        return {
            "message": "Database created successfully",
            "collection": collection,
            "path": path
        }
    except Exception as e:
        logger.error(f"Error on 'create': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Update an existing database/collection
@app.put("/collections/{collection}/{path:path}")
async def update_collection(
    collection: str, 
    path: str, 
    config: InfrangConfig = None
):
    
    if config is None:
        config = InfrangConfig()
    try:
        infrang = get_infrang_instance(collection, config)
        infrang.update(kb_path=path)
        return {
            "message": "Database updated successfully",
            "collection": collection,
            "path": path
        }
    except Exception as e:
        logger.error(f"Error on 'update': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Delete a collection
@app.delete("/collections/{collection}")
async def delete_collection(
    collection: str,
    config: InfrangConfig = None
):
    if config is None:
        config = InfrangConfig()
    try:
        infrang = get_infrang_instance(collection, config)
        infrang.delete()
        return {
            "message": "Collection deleted successfully",
            "collection": collection
        }
    except Exception as e:
        logger.error(f"Error on 'delete': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Perform semantic search and generate answer
@app.post("/answer/{collection}")
async def answer_query(
    collection: str,
    query: str = Query(..., description="The query for searching and answering"),
    config: InfrangConfig = None
):
    if config is None:
        config = InfrangConfig()
    
    try:
        infrang = get_infrang_instance(collection, config)
        result = infrang.answer(query=query)
        return {
            "collection": collection,
            "query": query,
            "result": result
        }
    except Exception as e:
        logger.error(f"Answer error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=7456)