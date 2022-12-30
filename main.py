from fastapi import FastAPI
from src.api import router
from src.utils.models import TrainModel

app = FastAPI()

app.include_router(router=router)

TrainModel()
