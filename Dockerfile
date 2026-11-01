FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y libgl1

RUN pip install flask tensorflow numpy opencv-python

EXPOSE 5000

CMD ["python","api/app.py"]