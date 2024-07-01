# Import the Speech-to-Text client library
from google.cloud import speech
from google.protobuf import wrappers_pb2
from google.cloud import storage
import requests

# Instantiates a client
client=speech.SpeechClient()
storage_client = storage.Client()

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    # Make the blob publicly readable
    blob.make_public()

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

    # Return the public URL
    return f"gs://{bucket_name}/{destination_blob_name}"


def transcribe_speech(gcs_uri):
  audio=speech.RecognitionAudio(gcs_uri)

  config=speech.RecognitionConfig(
  encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
  sample_rate_hertz=48000,
  language_code="ml-IN",
  model="latest_short",
  audio_channel_count=1,
  enable_word_confidence=True,
  enable_word_time_offsets=True,
  )

  # Detects speech in the audio file
  operation=client.long_running_recognize(config=config, audio=audio)

  print("Waiting for operation to complete...")
  response=operation.result(timeout=90)

  string = ""
  for result in response.results:
    string+result.alternatives[0].transcript
  
  response = requests.post("http://localhost:3000/translateText", json={"text": string, "lang": "en"})
  return response.json()

