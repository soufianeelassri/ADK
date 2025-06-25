import os
import json
from dotenv import load_dotenv
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import storage

load_dotenv()

def read_gcs_file(gs_path: str) -> str:
    try:
        if not gs_path.startswith("gs://"):
            return "(Lien non GCS invalide)"

        path_without_prefix = gs_path[5:]
        bucket_name, blob_path = path_without_prefix.split("/", 1)

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        content = blob.download_as_text(encoding="utf-8")
        return content

    except Exception as e:
        print(f"Erreur lecture GCS {gs_path}: {e}")
        return f"(Impossible de lire le contenu : {e})"

def retrieve_from_knowledge_base(query: str) -> list[dict]:
    client = discoveryengine.SearchServiceClient()
    serving_config = client.serving_config_path(
        project=PROJECT_ID,
        location=LOCATION,
        data_store=DATA_STORE_ID,
        serving_config="default_config",
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=5,
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=5,
                ignore_adversarial_query=True,
                include_citations=True,
            ),
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True
            ),
        ),
    )

    try:
        response = client.search(request)
        print(f"Successfully queried Vertex AI Search for: '{query}'")

        results = []
        for search_result in response.results:
            doc_data = search_result.document
            derived = doc_data.derived_struct_data or {}

            content = "(Aucun extrait disponible)"
            if "link" in derived and derived["link"].startswith("gs://"):
                content = read_gcs_file(derived["link"])
            else:
                if "extractive_segments" in derived and derived["extractive_segments"]:
                    content = derived["extractive_segments"][0]
                elif "content" in derived and derived["content"]:
                    content = derived["content"]

            results.append({
                "id": doc_data.id,
                "title": derived.get("title", "No Title"),
                "link": derived.get("link", ""),
                "content": content,
            })

        return results

    except Exception as e:
        print(f"An error occurred during Vertex AI Search query: {e}")
        return [{"error": f"Failed to query knowledge base: {e}"}]
