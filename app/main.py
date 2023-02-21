from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from .routers import fire, knife, arms, fight, emotions


api_app  = FastAPI(
    # middleware=middleware,
    title="Fire Detection API Service",
    description="""Fire Detection Service""",
    version="0.0.1",
    contact={
        "name": "Ali Amini Bagh",
        "email": "aliaminibagh@gmail.com",
    }
)

api_app.include_router(fire.router)
api_app.include_router(knife.router)
api_app.include_router(arms.router)
api_app.include_router(fight.router)
api_app.include_router(emotions.router)
# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


app = FastAPI(title="main app")
app.mount("/api", api_app)
app.mount("/", StaticFiles(directory="./ui", html=True), name="ui")


@api_app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url=f"/api/docs/", status_code=303)
