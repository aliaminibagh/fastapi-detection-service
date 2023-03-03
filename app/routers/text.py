from fastapi import APIRouter


router = APIRouter(
    prefix="",
    tags=["text"],
)

@router.post("/infer_text")
async def text_infer(text: str):
    return {"result": f'{text}'}