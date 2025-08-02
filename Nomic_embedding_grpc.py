# License
#
# This software is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
#
# Copyright (C) 2025 Roland Kohlhuber
#
# Note: The AI models used by this software (nomic-ai/nomic-embed-text-v2-moe, nomic-ai/nomic-embed-vision-v1.5) retain their original Apache 2.0 licenses and are not subject to the AGPL license terms.
#
# For the complete license text, see: https://www.gnu.org/licenses/agpl-3.0.html

# python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. embedding.proto


import asyncio
import base64
import re
import traceback
from io import BytesIO
from typing import List, Tuple

import aiohttp
import grpc
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel

# Import generated classes from your .proto file
import embedding_pb2
import embedding_pb2_grpc

# --- Model Loading ---
print("Loading models...")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
vision_model_name = "nomic-ai/nomic-embed-vision-v1.5"
vision_processor = AutoImageProcessor.from_pretrained(vision_model_name)
vision_model = AutoModel.from_pretrained(vision_model_name, trust_remote_code=True)

# Select the best available device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
vision_model.to(device)
print(f"Using device: {device}")

# Print available prompt types for debugging
print("Available prompt types:", list(model.prompts.keys()))

# --- Dynamic Batching Implementation ---

class DynamicBatcher:
    """
    Collects individual requests and processes them in batches. This elegantly
    handles the trade-off between latency and throughput by using two triggers:
    1. A timeout (for low-load responsiveness).
    2. A max batch size (for high-load throughput).
    """
    def __init__(self, model, max_batch_size=32, batch_timeout=0.1):
        self.model = model
        self.max_batch_size = max_batch_size  # The "hard cap" for batch size
        self.batch_timeout = batch_timeout    # The "dynamic timeframe" for low load
        self.queue = []
        self.lock = asyncio.Lock()
        self.batch_processor_task = None
        print(f"DynamicBatcher initialized: Max Batch Size={max_batch_size}, Timeout={batch_timeout*1000}ms")

    async def _batch_processor(self):
        """The background task that processes the queue based on the timeout."""
        print("Background batch processor started.")
        while True:
            await asyncio.sleep(self.batch_timeout)
            async with self.lock:
                if self.queue:
                    await self._process_queue()

    def _map_text_type_to_prompt(self, text_type_enum):
        """Map protobuf TextType enum to Nomic model prompt names"""
        # Get the enum name
        type_name = embedding_pb2.TextType.Name(text_type_enum).lower()
        
        # Map to valid Nomic prompt types
        prompt_mapping = {
            'query': 'query',
            'passage': 'passage',
            'classification': 'Classification',
            'multilabelclassification': 'MultilabelClassification',
            'clustering': 'Clustering',
            'pairclassification': 'PairClassification',
            'sts': 'STS',
            'summarization': 'Summarization',
            'speed': 'Speed'
        }
        
        # Return mapped prompt or default to 'query'
        prompt_name = prompt_mapping.get(type_name, 'query')
        print(f"Mapped TextType {type_name} to prompt '{prompt_name}'")
        return prompt_name

    async def _process_queue(self):
        """Processes the current items in the queue."""
        if not self.queue:
            return

        current_batch = self.queue[:]
        self.queue.clear()
        
        print(f"Processing batch of size {len(current_batch)}")

        # Group requests by their type ('query', 'passage', etc.) to process them correctly
        requests_by_type = {}
        for text_req, future in current_batch:
            embed_type = self._map_text_type_to_prompt(text_req.type)
            if embed_type not in requests_by_type:
                requests_by_type[embed_type] = {'texts': [], 'futures': []}
            requests_by_type[embed_type]['texts'].append(text_req.text)
            requests_by_type[embed_type]['futures'].append(future)

        # Process each group as a separate batch
        for embed_type, data in requests_by_type.items():
            try:
                print(f"Processing {len(data['texts'])} texts with prompt type '{embed_type}'")
                embeddings = self.model.encode(data['texts'], prompt_name=embed_type, device=device)
                # Return the result to each waiting request
                for i, future in enumerate(data['futures']):
                    if not future.done():
                        future.set_result(embeddings[i])
            except Exception as e:
                print(f"Error processing batch for type {embed_type}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                for future in data['futures']:
                    if not future.done():
                        future.set_exception(e)

    async def add_request(self, text_req):
        """Adds a request to the batch and returns a future for the result."""
        future = asyncio.Future()
        
        async with self.lock:
            self.queue.append((text_req, future))
            
            # This is the "hard cap" trigger. If the batch is full, process it
            # immediately without waiting for the timeout.
            if len(self.queue) >= self.max_batch_size:
                await self._process_queue()
        
        return await future

    def start(self):
        """Starts the background batch processor task."""
        if self.batch_processor_task is None:
            self.batch_processor_task = asyncio.create_task(self._batch_processor())

    async def stop(self):
        """Stops the background task and processes any remaining items."""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
            print("Batch processor stopped. Processing final queue...")
            await self._process_queue()
            self.batch_processor_task = None


# --- Helper Functions (Unchanged from original) ---

def get_image_embedding(image):
    inputs = vision_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = vision_model(**inputs).last_hidden_state
    return F.normalize(img_emb[:, 0], p=2, dim=1).cpu().numpy()[0]

def get_image_embeddings(images: List[Image.Image]):
    inputs = vision_processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = vision_model(**inputs).last_hidden_state
    return F.normalize(img_emb[:, 0], p=2, dim=1).cpu().numpy()

async def get_image_from_data(image_data: str, context):
    if image_data.startswith('http'):
        # Detect local IPFS URLs and rewrite to public gateway
        if 'localhost:8080/ipfs/' in image_data:
            # Extract CID (e.g., Qme9gsAZxKvcYxhiZSsWGiM...)
            match = re.search(r'/ipfs/(Qm[0-9a-zA-Z]+)', image_data)
            if match:
                cid = match.group(1)
                image_data = f'https://ipfs2.ome.sh/ipfs/{cid}'  # Or use dweb.link, cloudflare-ipfs.com, etc.
                print(f"Rewrote local IPFS URL to public: {image_data}")
        
        # Disable SSL verification for all image fetches (insecure, use only if necessary)
        print("Warning: SSL verification disabled for image fetch (insecure).")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_data, ssl=False) as response:
                    if response.status == 200:
                        content = await response.read()
                        return Image.open(BytesIO(content)).convert("RGB")
                    else:
                        print(f"Failed to fetch image from {image_data}: HTTP {response.status}")
                        return None
        except Exception as e:
            print(f"Error fetching image from {image_data}: {str(e)}")
            return None
    else:
        try:
            image_bytes = base64.b64decode(image_data)
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"Invalid base64 image data: {str(e)}")
            return None


# --- gRPC Servicer Implementation ---

# Instantiate the batcher. This single instance will be shared across all requests.
text_batcher = DynamicBatcher(model)

class EmbeddingServiceServicer(embedding_pb2_grpc.EmbeddingServiceServicer):
    
    async def Embed(self, request: embedding_pb2.EmbedRequest, context: grpc.aio.ServicerContext):
        try:
            content_type = request.WhichOneof('content')
            if content_type == 'text_request':
                # The 'Embed' endpoint now uses the batcher for high performance
                text_req = request.text_request
                print(f"Received text embedding request: '{text_req.text[:50]}...' with type {embedding_pb2.TextType.Name(text_req.type)}")
                
                full_embedding = await text_batcher.add_request(text_req)
                
                return embedding_pb2.EmbeddingResponse(
                    embedding=embedding_pb2.EmbeddingVector(values=full_embedding.tolist()),
                    model='nomic-ai/nomic-embed-text-v2-moe',
                    dimensions=len(full_embedding)
                )

            elif content_type == 'image_request':
                # Image embedding could also be batched using a similar pattern
                image_req = request.image_request
                print(f"Received image embedding request: {image_req.image_data[:50]}...")
                
                image = await get_image_from_data(image_req.image_data, context)
                if image is None:
                    full_embedding = np.zeros(768)
                else:
                    full_embedding = get_image_embedding(image)

                return embedding_pb2.EmbeddingResponse(
                    embedding=embedding_pb2.EmbeddingVector(values=full_embedding.tolist()),
                    model=vision_model_name,
                    dimensions=len(full_embedding)
                )
            else:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Request must contain either text_request or image_request.')

        except Exception as e:
            trace = traceback.format_exc()
            print(f"Error in Embed: {e}\n{trace}")
            await context.abort(grpc.StatusCode.INTERNAL, f"An internal error occurred: {str(e)}")

    # The explicit batch endpoints are still useful for clients that want to send pre-batched data.
    async def EmbedTextBatch(self, request: embedding_pb2.TextBatchRequest, context: grpc.aio.ServicerContext):
        try:
            embed_type = embedding_pb2.TextType.Name(request.type).lower()
            # Map to valid prompt name
            prompt_mapping = {
                'query': 'query',
                'passage': 'passage',
                'classification': 'Classification',
                'multilabelclassification': 'MultilabelClassification',
                'clustering': 'Clustering',
                'pairclassification': 'PairClassification',
                'sts': 'STS',
                'summarization': 'Summarization',
                'speed': 'Speed'
            }
            prompt_name = prompt_mapping.get(embed_type, 'query')
            
            embeddings = model.encode(list(request.texts), prompt_name=prompt_name, device=device)
            embedding_vectors = [embedding_pb2.EmbeddingVector(values=emb.tolist()) for emb in embeddings]
            return embedding_pb2.BatchEmbeddingResponse(embeddings=embedding_vectors, model='nomic-ai/nomic-embed-text-v2-moe', dimensions=embeddings.shape[1])
        except Exception as e:
            print(f"Error in EmbedTextBatch: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def EmbedImageBatch(self, request: embedding_pb2.ImageBatchRequest, context: grpc.aio.ServicerContext):
        try:
            images = []
            for img_data in request.images:
                image = await get_image_from_data(img_data, context)
                images.append(image)
            
            embeddings_np = np.zeros((len(images), 768))
            
            valid_images = [img for img in images if img is not None]
            if valid_images:
                valid_embs = get_image_embeddings(valid_images)
                valid_idx = 0
                for i, img in enumerate(images):
                    if img is not None:
                        embeddings_np[i] = valid_embs[valid_idx]
                        valid_idx += 1
            
            embedding_vectors = [embedding_pb2.EmbeddingVector(values=emb.tolist()) for emb in embeddings_np]
            return embedding_pb2.BatchEmbeddingResponse(embeddings=embedding_vectors, model=vision_model_name, dimensions=768)
        except Exception as e:
            print(f"Error in EmbedImageBatch: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


# --- Server Startup ---

async def serve():
    server = grpc.aio.server()
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(EmbeddingServiceServicer(), server)
    
    # Try different ports if 60051 is in use
    ports_to_try = [60051, 60052, 60053]
    listen_addr = None
    
    for port in ports_to_try:
        try:
            addr = f'[::]:{port}'
            server.add_insecure_port(addr)
            listen_addr = addr
            break
        except Exception as e:
            print(f"Failed to bind to port {port}: {e}")
            continue
    
    if not listen_addr:
        print("Failed to bind to any port. Exiting.")
        return
    
    # Start the dynamic batcher's background processing task
    text_batcher.start()
    
    print("Models loaded, server and batcher started...")
    print(f"Listening on {listen_addr}")
    
    try:
        await server.start()
        await server.wait_for_termination()
    finally:
        # Ensure graceful shutdown of the server and the batcher
        print("Shutting down server...")
        await server.stop(0)
        print("Stopping batcher...")
        await text_batcher.stop()
        print("Shutdown complete.")


if __name__ == '__main__':
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("\nServer stopping by KeyboardInterrupt...")
