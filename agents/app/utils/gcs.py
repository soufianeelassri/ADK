import logging
from google.cloud import storage
from google.api_core import exceptions

def create_bucket_if_not_exists(bucket_name: str, project_id: str, location: str):
    """Creates a GCS bucket if it does not already exist."""
    
    if bucket_name.startswith("gs://"):
        bucket_name = bucket_name[5:]

    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)

        if not bucket.exists():
            logging.info(f"Staging bucket '{bucket_name}' not found. Creating it in {location}...")
            bucket.create(location=location)
            logging.info(f"Bucket '{bucket_name}' created successfully.")
        else:
            logging.info(f"Using existing staging bucket: {bucket_name}")
            
    except exceptions.GoogleAPICallError as e:
        logging.error(f"Failed to create or access GCS bucket '{bucket_name}': {e}")
        raise