# Wildfire Prediction Streamlit App

This repository contains a Streamlit app that predicts wildfire risk based on weather and location data.

## Run locally

1. python -m venv .venv
2. .venv\Scripts\activate
3. pip install -r requirements.txt
4. streamlit run fireapp.py

## Deploy to Streamlit Community Cloud (free)

1. Push repository to GitHub.
2. Go to https://streamlit.io/cloud, sign in with GitHub.
3. Create a new app, select this repository, branch `main`, file `fireapp.py`.
4. In app settings -> Secrets, add:

```toml
WEATHER_API_KEY = "your-weatherapi-key"
OPENWEATHERMAP_API_KEY = "your-openweathermap-key"
OPENCAGE_API_KEY = "your-opencage-key"
```

5. Deploy and open the URL.

## Notes

- Model and pipeline files must be present in repository:
  - preprocessing_pipeline_cls_last.pkl
  - dnn_best_params.pkl
  - dnn_model_last.pth

- If model is large (>100MB), use Git LFS or cloud storage and update `fireapp.py` file path accordingly.
