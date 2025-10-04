import multiprocessing

import uvicorn
from decouple import config
from fastapi import FastAPI


from routes.search import router 


# =============================================================================

FASTAPI_HOST = config("FASTAPI_HOST")
FASTAPI_PORT = config("FASTAPI_PORT", cast=int)
FASTAPI_RELOAD = config("FASTAPI_RELOAD", cast=bool)

# =============================================================================

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    title="Smart Journalist Project",
    description="""<img src='/static/swagger-ui/neshan-dark-back.svg'
    width='200'/><br><br>
    Smart Journalist Project is an access management system for
    AI-driven agents, developed by IranSamaneh 🏢.

This project is built using the FastAPI framework ⚙️ and provides
administrators with the ability to register, log in, generate API keys 🔐,
and manage requests related to news-generation agents 📰.

This system is designed to ensure secure access control, efficient tracking,
and streamlined operation of AI-powered journalism tools 🤖.
""",
    version="0.0.1",
)


app.include_router(router, prefix="", tags=["Ticket"])


if __name__ == "__main__":

    if FASTAPI_RELOAD:
        uvicorn.run(
            "main:app",
            host=FASTAPI_HOST,
            port=FASTAPI_PORT,
            reload=FASTAPI_RELOAD
        )

    else:
        cpu_count = multiprocessing.cpu_count()
        uvicorn.run(
            "main:app",
            host=FASTAPI_HOST,
            port=FASTAPI_PORT,
            workers=4  # cpu_count
        )