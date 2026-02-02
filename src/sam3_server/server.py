import os
import shutil
import uvicorn
import argparse
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from sam3_server.inference import SAM3Inference
from sam3_server.schemas import InferenceRequest

class Sam3Server:
    def __init__(self, checkpoint_path: str, upload_dir: str, cache_dir: str):
        self.app = FastAPI(title="Sam3 Server")
        self.upload_dir = upload_dir
        self.cache_dir = cache_dir
        self.sam = SAM3Inference(
            checkpoint_path=checkpoint_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        for path in [self.upload_dir, self.cache_dir]:
            os.makedirs(path, exist_ok=True)

        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/upload")
        async def upload_image(file: UploadFile = File(...)):
            return await self.handle_upload(file)

        @self.app.get("/download/{filename}")
        async def download_image(filename: str):
            return await self.handle_download(filename)

        @self.app.get("/status")
        async def get_status():
            return {
                "upload_dir": self.upload_dir,
                "cache_dir": self.cache_dir,
                "files_stored": len(os.listdir(self.upload_dir))
            }

        @self.app.post("/inference")
        async def post_inference(request: InferenceRequest):
            return await self.handle_inference(request)

    async def handle_upload(self, file: UploadFile):
        file_path = os.path.join(self.upload_dir, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
        return {"filename": file.filename, "status": "success"}

    async def handle_download(self, filename: str):
        file_path = os.path.join(self.upload_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)

    async def handle_inference(self, request: InferenceRequest):
        results = []

        for path in request.filepaths:

            filepath = os.path.join(self.upload_dir, path)
            if not os.path.exists(filepath):
                results.append({"path": filepath, "error": "File not found"})
                continue

            try:
                with open(filepath, "rb") as f:
                    img_bytes = f.read()

                inference_result = self.sam.run_inference(img_bytes, request.prompt)
                results.append({"path": path, "result": inference_result})
                print(inference_result)

            except Exception as e:
                results.append({"path": path, "error": str(e)})

        return {"batch_results": results}

def main():
    parser = argparse.ArgumentParser(description="FastAPI Image Server with custom paths.")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/opt/weights/sam3.pt",
        help="Path to the model weights"
    )
    parser.add_argument(
        "--uploads",
        type=str,
        default="/tmp/sam3/uploads",
        help="Directory to store uploaded images"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="/tmp/sam3/cache",
        help="Directory for temporary cache data"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on"
    )

    args = parser.parse_args()

    # Initialize and run
    server = Sam3Server(
        checkpoint_path=args.checkpoint,
        upload_dir=args.uploads,
        cache_dir=args.cache)

    print(f"Starting server...")
    print(f"Uploads dir: {os.path.abspath(args.uploads)}")
    print(f"Cache dir:   {os.path.abspath(args.cache)}")

    uvicorn.run(server.app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()