from flask import Flask, request, jsonify, send_file
from google.cloud import storage, vision
import os
import pandas as pd
import uuid

app = Flask(__name__)

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path_to_your_service_account.json'

# Initialize Google Cloud clients
storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()

# Cloud Storage bucket for temporary file storage
BUCKET_NAME = 'your-gcs-bucket-name'

@app.route('/upload', methods=['POST'])
def upload_images():
    """Handles image uploads and starts processing."""
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    images = request.files.getlist('images')
    image_urls = []

    # Upload images to GCS
    bucket = storage_client.bucket(BUCKET_NAME)
    for image in images:
        blob_name = f"images/{uuid.uuid4()}_{image.filename}"
        blob = bucket.blob(blob_name)
        blob.upload_from_file(image)
        image_urls.append(f"gs://{BUCKET_NAME}/{blob_name}")

    return jsonify({'message': 'Images uploaded successfully', 'image_urls': image_urls}), 200

@app.route('/process', methods=['POST'])
def process_images():
    """Processes images and generates a CSV file."""
    data = request.json
    image_urls = data.get('image_urls')

    if not image_urls:
        return jsonify({'error': 'No image URLs provided'}), 400

    results = []

    for image_url in image_urls:
        blob = storage.Blob.from_string(image_url, client=storage_client)
        image_content = blob.download_as_bytes()

        # Use Vision API to extract text
        image = vision.Image(content=image_content)
        response = vision_client.text_detection(image=image)

        if response.error.message:
            return jsonify({'error': response.error.message}), 500

        text = response.full_text_annotation.text
        results.append({'image_url': image_url, 'extracted_text': text})

    # Convert results to CSV
    df = pd.DataFrame(results)
    csv_file = f"output_{uuid.uuid4()}.csv"
    df.to_csv(csv_file, index=False)

    # Upload CSV to GCS
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"outputs/{csv_file}")
    blob.upload_from_filename(csv_file)

    csv_url = f"gs://{BUCKET_NAME}/outputs/{csv_file}"

    return jsonify({'message': 'Processing complete', 'csv_url': csv_url}), 200

@app.route('/download', methods=['GET'])
def download_csv():
    """Downloads the processed CSV file."""
    csv_url = request.args.get('csv_url')

    if not csv_url:
        return jsonify({'error': 'CSV URL not provided'}), 400

    blob = storage.Blob.from_string(csv_url, client=storage_client)
    csv_file = csv_url.split('/')[-1]
    blob.download_to_filename(csv_file)

    return send_file(csv_file, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
