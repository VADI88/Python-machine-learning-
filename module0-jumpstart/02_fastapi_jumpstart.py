# LIBRARY

import pandas as pd
import pycaret.classification as clf
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# Load the trained model

model = clf.load_model(model_name="models/js_xgb_model_finalized")

app = FastAPI(name="Email - Lead scoring predictions")


@app.get("/")  # root URL
async def main():
    content = """
    <body>
    <h1> Welcome to EMAIL Lead scoring project. </h1>
    <p> Navigate to <code>/docs</code> to see the API Documentation</p>
    </body>
    """

    return HTMLResponse(content=content)


@app.post("/predict")
async def predict(member_rating: int, country_code: str):
    new_df = pd.DataFrame(
        dict(
            member_rating=[member_rating],
            country_code=[country_code],
        )
    )

    predictions = clf.predict_model(
        estimator=model, data=new_df, raw_score=True
    )

    return JSONResponse(predictions.to_dict(orient="records"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
