from fastapi import FastAPI, Header
from fastapi.exceptions import HTTPException
import uvicorn
from contextlib import asynccontextmanager
from dreamsculpt_be.models.generation_request import GenerationRequest
from dreamsculpt_be.models.generation_response import GenerationResponse
from dreamsculpt_be.inference_core.scheduler import scheduler_loop
from dreamsculpt_be.utils.logging import publish_logs
import asyncio
from multiprocessing import Queue, Process, set_start_method
from typing import Dict
import uuid
import boto3

request_tracker: Dict[str, asyncio.Future] = {}
s3_client = boto3.client('s3')

# --- Listen and process completed requests from the scheduler process ---
async def result_listener(result_queue: Queue):
    loop = asyncio.get_running_loop()
    while True:
        # Run in seperate thread since queue.get() is blocking
        request_id, response = await loop.run_in_executor(None, result_queue.get)
        # Resolve the future and remove from tracker
        future = request_tracker.pop(request_id, None)
        if future is not None and not future.done():
            if response["error"]:
                future.set_exception(HTTPException(status_code=response["error"]["status_code"], detail=response["error"]["detail"]))
            else:
                future.set_result(response["result"])


# --- Startup tasks: Start scheduler and listener ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize scheduler and setup IPC
    app.state.ipc_request_queue = Queue()
    app.state.ipc_result_queue = Queue()
    scheduler_process = Process(target=scheduler_loop, args=[app.state.ipc_request_queue, app.state.ipc_result_queue])
    scheduler_process.start()
    asyncio.create_task(result_listener(app.state.ipc_result_queue))
    yield
    print("Shutting Down...", flush=True)


app = FastAPI(lifespan=lifespan)


# --- Endpoints ---
@app.get("/health")
def health_check():
    return "System is healthy!"


@app.post("/generate", response_model = GenerationResponse)
async def generate(request: GenerationRequest, session_id: uuid.UUID = Header()) -> str:
    # Generate request ID + Future, add to req tracker
    request_id = str(uuid.uuid4())
    request_future = asyncio.get_running_loop().create_future()
    request_tracker[request_id] = request_future

    # Dispatch request to scheduler
    app.state.ipc_request_queue.put((session_id, request_id, request.image_prompt, request.text_prompt))
    print("request added to queue", flush=True)

    # Return the generated image
    generated_image: str = await request_future
    print("result generated", flush=True)
    publish_logs(s3_client, request_id, request.text_prompt, request.image_prompt, generated_image)
    return {"generated_image": generated_image}


# --- Server Start ---
def main():
    set_start_method("spawn", force=True)
    uvicorn.run("dreamsculpt_be.main:app", host="0.0.0.0")


if __name__ == "__main__":
    main()

"""
09/05/2025

Decided to host Flux Schnell in EC2 due to gemini rate limits
Preliminary stack: FastAPI + huggingface for inference server
    - Refer to fast-flux to optimize inference
Using UV for project management
    - Discovered that specifying a build backend causes uv to enforce stricter folder structure (src/package_name/)
    - Discovered that I cant specify custom pyproject.toml cli commands unless the package installed globally
        - wanted to have "uv run lint" trigger a ruff linter check

09/12/2025

Set up folder structure
    - src/dreamsculpt_be contains the source code
        - UV requires src/ because I specified a build backend in pyproject.toml, meaning this repo can be packaged and distributed
Setup uv run start command
    - the pyproject.toml section is called [project.scripts] and the kv pair is "<command>": "dreamsculpt_be.main:main"
        - the . is because dreamsculpt_be is a python module. 
    - main() starts the server
Catching up on how diffusion models work, dont wanna implement something I dont understand. Will start with a script for single image gen
    - Diffusion:
        - Noising: Iteratively add noise to an input image, noise follows Guassian distribution
        - De-noising: Given an noised image for a given timestep, the model predicts the noise added at that timestep.
    - Flux:
        - De-noising: Predict a vector space position diff (delta_x(t), AKA v(t) ) during each generation step.

09/15/2025
Scaffolded /generate endpoint: Am able to return a base64 encoded dummy PIL image. I also setup base inference script. Next step is to containerize, and deploy

09/17/2025
Chose a pytorch image from dockerhub, and setup my dockerfile. Need to get container build to work properly locally

09/19/2025

Container builds locally. Verfied E2E flow with DrawThings server. Need to fix preview contraction bug where offset does not persist

10/02/2025
    Working to get single image gen working on ec2. Tried with g5g.xlarge but ran out of system memory. Trying again with g5.xlarge

    
12/24/2025
    Finished optimizing AI inference configs in my test script and started working on server code. Endpoints are setup, schema is defined,
    and concurrency model is mostly fleshed out. Server will queue up requests, the scheduler will dequeue, batch, and trigger inference.
    The queue contains a tuple of a Future and the input image. The scheduler resolves the future when generation is complete, and
    FastAPI will return the result immediately since the future is awaited.

    I have a problem. Incoming requests wont get queued up until the current image generation is complete. I need to have a seperate process
    to avoid blocking the main event loop. But this opens up an IPC problem! I'd need to open 2 queues between the main and secondary processes:
        1 for each direction of communication. 

    Okay, I redid the concurrency model. Used a Pipe to communicate with the child scheduler process. Implemented batching and a global request tracker.

12/26/2025
    Lowkey annoyed, docker keeps downloading torch even though its already provided by the base image. UV doesnt allow me to install deps from a lockfile outside of a virtualenv. I woudl have to export to a requirements.txt first, dont want to do that workaround. 
    Tried setting the project env to the system environment but its till redundently downloading torch, thus dramatically increasing build time.

12/28/2025
    Alright, I caved and created a requirements.txt so that I can install deps using uv's "pip" interface, which allows you to install in the system env. Its still downloading torch ugh. 
    Lets just get the container working and deployed, I'll optimize the container later. 

    Wait what? I was getting an OOM error during app startup, turns out colima was consuming 80GB of disk space. I totally wiped and restarted colima, and now my previous issue is 
    no longer occuring! Its not re-downloading torch anymore!!! Not sure why or how this fixed it. OH WAIT, ITS BC I SWITCHED TO AN IMAGE THAT USES THE SAME PYTORCH VERSION AS IN MY LOCKFILE! I think
    uv noticed that the torch version in the lockfile differed from that in the base image, which triggered the re-download!

    LFG! I am able to start the container locally and hit the /generate and /health endpoints!
    DreamSculpt Deployment:
        1) Convert image into a tar
            - docker save -o dreamsculpt-0.0.9.tar dreamsculpt:0.0.9 
        2) scp to ec2 instance:
            - scp -i "Rahul Key Pair.pem" dreamsculpt-0.0.9.tar ec2-user@13.221.123.53:/home/ec2-user
            - ~13GB image, this takes like 10 minutes :((
        3) ssh into instance and load image:
            - docker load -i dreamsculpt-0.0.9.tar
        4) Start container:
            - docker run -e HF_TOKEN=<HUGGINGFACE TOKEN> -p 80:8000 --gpus all <image_id>
            - docker run -e GEMINI_API_KEY=<GEMINI TOKEN> -p 8000:8000 <image_id
    
    Log Viewer Deployment:
        1) docker build -t dreamsculpt-log-viewer:latest -f Dockerfile.log-viewer .
        2) docker save -o dreamsculpt-log-viewer-0.0.1.tar
        3) scp -i "Rahul Key Pair.pem" dreamsculpt-log-viewer-0.0.1.tar ec2-user@13.221.123.53:/home/ec2-user
        3) docker run -p 8501:8501 <image_id>

    Okay, now lets connect the server to the AI inference pipeline and redeploy

12/30/2025
    Alright model init is failing, i think because the gpu isnt visible to the container?

1/1/2026
    Added some missing runtime dependencies (protobuf, sentencepeice) to resolve model init failure, and enabled gpu passthrough on the container.

1/2/2025
    Great news, the server works! Bad news, server deadlocks on burst load. Turns out pipes have a 64kB limit, which means I cant have more that 2-3 images queued up.
    Split scheduler into 2 threads: one to constantly drain the pipe into the Queue and one to drain the queue + trigger AI inference
    My current single threaded scheduler has another critial flaw, where AI inference cannot trigger until the pipe is fully drained. 

    - Fixed a frontend bug where the generated image preview wont render
            - Quotes around the base64 image string broke rendering
    - Fixed image preview dragging bug
    - Decoupled IPC listener from main scheduler thread to avoid blocking dispatch. 

1/4/2025
    App works end to end. Generation takes 8.7 seconds, additional optimizations will get it down to 7.5. I notice that generation quality isnt the best - stick figures dont get rendered into acual people. Need to tune prompt

1/18/2026
    Added session de-duping in the queue. Now, a max of 1 request per session will be in the queue at any given time. Added a mandatory sessionId header as well


2/21/2026
    - Added Gemini support

3/1/2026
    - Added Generate button to frontend.

3/8/2026
    - Server side generation limit enforcement
    - Enhance error handling to propagate scheduler exceptions back to main process

3/11/2026
    - Reworked dependency groups to try to slim down image size when using gemini. Hopefully I can deploy to a smaller instance size
    - Ugh. Ok use am arm64 instance. I accidentally spun up an amd64 instance. 

3/14/2026
    - Fixed some import issues because I removed pytorch from the container. 


3/18/2026

    - HOLY FUCK!
    - Used sslip to assign my ec2 instance a domain name thats its IP + ".sslip.io"
    - This allowed me to use caddy to create an HTTP termination proxy with automatic SSL certificate generation
    - Downloaded Dreamsculpt onto my Iphone!

    - Bugs:
        - Tool picker isnt aligned to the bottom of the screen
        - Make the prompt bar auto-collapse to minimize canvas occlusion
        - Set gemini price limit for this API key

To Do:
    - Investigate why CPU usage is so high.
"""
