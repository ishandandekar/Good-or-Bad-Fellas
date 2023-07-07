FROM python:3.9

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./api ./api

COPY ./models ./models

# CMD ["uvicorn", "api.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
CMD [ "python", "./api/serve_model.py" ]