from pydantic import Field
from pydantic import BaseModel as PydanticBaseModel, Field
from prehealthdata import PreHealthDataOutput
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


class StressPercentage(PydanticBaseModel):
    stress: float = Field(description="The stress percentage of the user")


@app.post("/healthdata")
async def health_data(stress: StressPercentage):
    stress = stress.stress
    result = PreHealthDataOutput(stress)
    return {
        "isStressed": result["isStressed"],
        "percentStressed": result["percentStressed"],
        "stressLevel": result["stressLevel"],
        "remedies": result["remedies"]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
