from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.connection import close_mongo_connection, connect_to_mongo, get_database
from app.routers.admin import router as admin_router
from app.routers.auth import router as auth_router
from app.routers.summarization import router as summarization_router
from app.routers.evaluation import router as evaluation_router
from app.routers.history import router as history_router


@asynccontextmanager
async def lifespan(app: FastAPI):

    await connect_to_mongo()
    try:
        yield
    finally:
        await close_mongo_connection()


app = FastAPI(title="FastAPI Auth with MongoDB", lifespan=lifespan)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên set cụ thể domain frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(summarization_router)
app.include_router(evaluation_router)
app.include_router(history_router)


@app.get("/")
async def root():

    db = get_database()
    collections = await db.list_collection_names()
    return {"message": "Connected to MongoDB!", "collections": collections}


