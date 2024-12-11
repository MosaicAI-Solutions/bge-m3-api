#!/bin/bash
# Test Script for Mixed Endpoints: /embeddings/ and /rerank/

# Check if the number of requests was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <num_requests>"
    exit 1
fi

# Configuration
url_embeddings="http://0.0.0.0:8000/embeddings/"
url_rerank="http://0.0.0.0:8000/rerank/"
num_requests=$1  # Number of concurrent requests

# JSON payloads for embeddings and rerank
json_embeddings='{
  "sentences": [
    "This is the first example sentence. Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.", "Sentence 2 from pair 1 Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum. ",
    "Here goes the second example. Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.", "Sentence 2 from pair 1 Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
  ]
}'

json_rerank='{
  "sentence_pairs": [
    ["Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.", "Lorem Ipsum is simply dummy text of the printing and typesetting industry. It has survived not only five centuries but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages.", "Lorem Ipsum is simply dummy text of the printing and typesetting industry. It has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.", "Lorem Ipsum passages and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages."],
    ["Lorem Ipsum is simply dummy text of the printing and typesetting industry. It has been the industrys standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.", "Lorem Ipsum passages and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages.", "When an unknown printer took a galley of type and scrambled it to make a type specimen book, Lorem Ipsum has been the industrys standard dummy text ever since the 1500s. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s. It has survived not only five centuries but also the leap into electronic typesetting, remaining essentially unchanged.", "Lorem Ipsum passages and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages."],
    ["When an unknown printer took a galley of type and scrambled it to make a type specimen book, Lorem Ipsum has been the industrys standard dummy text ever since the 1500s. Lorem Ipsum has been the industrys standard dummy text ever since the 1500s. It has survived not only five centuries but also the leap into electronic typesetting, remaining essentially unchanged.", "Lorem Ipsum passages and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages.", "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.", "Lorem Ipsum is simply dummy text of the printing and typesetting industry. It has survived not only five centuries but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages."]
  ]
}'


# Record the start time
start_time=$(date +%s)
echo "Testing mixed /embeddings/ and /rerank/ endpoints with $num_requests requests."

# Send requests
for i in $(seq 1 $num_requests); do
  if [ $((RANDOM % 2)) -eq 0 ]; then
    # Send to /embeddings/ endpoint
    curl -X POST "$url_embeddings" \
      -H "accept: application/json" \
      -H "Content-Type: application/json" \
      -H "X-API-Key: secure-api-key" \
      -d "$json_embeddings" &
  else
    # Send to /rerank/ endpoint
    curl -X POST "$url_rerank" \
      -H "accept: application/json" \
      -H "Content-Type: application/json" \
      -H "X-API-Key: secure-api-key" \
      -d "$json_rerank" &
  fi
done

# Wait for all requests to complete
wait
echo "All requests have been sent."

# Record the end time
end_time=$(date +%s)

# Calculate the duration
duration=$((end_time - start_time))
echo "Total time taken: $duration seconds"
