import os
import sys
import asyncio
import traceback

import nodes
import folder_paths
import execution
import uuid
import urllib
import json
import glob
import struct
import ssl
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from io import BytesIO

import aiohttp
from aiohttp import web
import logging

import mimetypes
from comfy.cli_args import args
import comfy.utils
import comfy.model_management
import node_helpers
from app.frontend_management import FrontendManager
from app.user_manager import UserManager
from model_filemanager import download_model, DownloadModelStatus
from typing import Optional
from api_server.routes.internal.internal_routes import InternalRoutes

import base64
import boto3

from botocore.exceptions import NoCredentialsError, ClientError

AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
AWS_BUCKET_NAME = ''
AWS_REGION = ''

class BinaryEventTypes:
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2

async def send_socket_catch_exception(function, message):
    try:
        await function(message)
    except (aiohttp.ClientError, aiohttp.ClientPayloadError, ConnectionResetError) as err:
        logging.warning("send error: {}".format(err))

@web.middleware
async def cache_control(request: web.Request, handler):
    response: web.Response = await handler(request)
    if request.path.endswith('.js') or request.path.endswith('.css'):
        response.headers.setdefault('Cache-Control', 'no-cache')
    return response

def create_cors_middleware(allowed_origin: str):
    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            # Pre-flight request. Reply successfully:
            response = web.Response()
        else:
            response = await handler(request)

        response.headers['Access-Control-Allow-Origin'] = allowed_origin
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, PUT, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    return cors_middleware

class PromptServer():
    def __init__(self, loop):
        PromptServer.instance = self

        mimetypes.init()
        mimetypes.types_map['.js'] = 'application/javascript; charset=utf-8'

        self.user_manager = UserManager()
        self.internal_routes = InternalRoutes()
        self.supports = ["custom_nodes_from_web"]
        self.prompt_queue = None
        self.loop = loop
        self.messages = asyncio.Queue()
        self.client_session:Optional[aiohttp.ClientSession] = None
        self.number = 0

        middlewares = [cache_control]
        if args.enable_cors_header:
            middlewares.append(create_cors_middleware(args.enable_cors_header))

        max_upload_size = round(args.max_upload_size * 1024 * 1024)
        self.app = web.Application(client_max_size=max_upload_size, middlewares=middlewares)
        self.sockets = dict()
        self.web_root = (
            FrontendManager.init_frontend(args.front_end_version)
            if args.front_end_root is None
            else args.front_end_root
        )
        logging.info(f"[Prompt Server] web root: {self.web_root}")
        routes = web.RouteTableDef()
        self.routes = routes
        self.last_node_id = None
        self.client_id = None

        self.on_prompt_handlers = []

        @routes.get('/ws')
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            sid = request.rel_url.query.get('clientId', '')
            if sid:
                # Reusing existing session, remove old
                self.sockets.pop(sid, None)
            else:
                sid = uuid.uuid4().hex

            self.sockets[sid] = ws

            try:
                # Send initial state to the new client
                await self.send("status", { "status": self.get_queue_info(), 'sid': sid }, sid)
                # On reconnect if we are the currently executing client send the current node
                if self.client_id == sid and self.last_node_id is not None:
                    await self.send("executing", { "node": self.last_node_id }, sid)

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        logging.warning('ws connection closed with exception %s' % ws.exception())
            finally:
                self.sockets.pop(sid, None)
            return ws

        @routes.get("/")
        async def get_root(request):
            response = web.FileResponse(os.path.join(self.web_root, "index.html"))
            response.headers['Cache-Control'] = 'no-cache'
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        @routes.get("/embeddings")
        def get_embeddings(self):
            embeddings = folder_paths.get_filename_list("embeddings")
            return web.json_response(list(map(lambda a: os.path.splitext(a)[0], embeddings)))

        @routes.get("/models/{folder}")
        async def get_models(request):
            folder = request.match_info.get("folder", None)
            if not folder in folder_paths.folder_names_and_paths:
                return web.Response(status=404)
            files = folder_paths.get_filename_list(folder)
            return web.json_response(files)

        @routes.get("/extensions")
        async def get_extensions(request):
            files = glob.glob(os.path.join(
                glob.escape(self.web_root), 'extensions/**/*.js'), recursive=True)

            extensions = list(map(lambda f: "/" + os.path.relpath(f, self.web_root).replace("\\", "/"), files))

            for name, dir in nodes.EXTENSION_WEB_DIRS.items():
                files = glob.glob(os.path.join(glob.escape(dir), '**/*.js'), recursive=True)
                extensions.extend(list(map(lambda f: "/extensions/" + urllib.parse.quote(
                    name) + "/" + os.path.relpath(f, dir).replace("\\", "/"), files)))

            return web.json_response(extensions)

        def get_dir_by_type(dir_type):
            if dir_type is None:
                dir_type = "input"

            if dir_type == "input":
                type_dir = folder_paths.get_input_directory()
            elif dir_type == "temp":
                type_dir = folder_paths.get_temp_directory()
            elif dir_type == "output":
                type_dir = folder_paths.get_output_directory()

            return type_dir, dir_type

        def compare_image_hash(filepath, image):
            hasher = node_helpers.hasher()
            
            # function to compare hashes of two images to see if it already exists, fix to #3465
            if os.path.exists(filepath):
                a = hasher()
                b = hasher()
                with open(filepath, "rb") as f:
                    a.update(f.read())
                    b.update(image.file.read())
                    image.file.seek(0)
                    f.close()
                return a.hexdigest() == b.hexdigest()
            return False

        def image_upload(post, image_save_function=None):
            image = post.get("image")
            overwrite = post.get("overwrite")
            image_is_duplicate = False

            image_upload_type = post.get("type")
            upload_dir, image_upload_type = get_dir_by_type(image_upload_type)

            if image and image.file:
                filename = image.filename
                if not filename:
                    return web.Response(status=400)

                subfolder = post.get("subfolder", "")
                full_output_folder = os.path.join(upload_dir, os.path.normpath(subfolder))
                filepath = os.path.abspath(os.path.join(full_output_folder, filename))

                if os.path.commonpath((upload_dir, filepath)) != upload_dir:
                    return web.Response(status=400)

                if not os.path.exists(full_output_folder):
                    os.makedirs(full_output_folder)

                split = os.path.splitext(filename)

                if overwrite is not None and (overwrite == "true" or overwrite == "1"):
                    pass
                else:
                    i = 1
                    while os.path.exists(filepath):
                        if compare_image_hash(filepath, image): #compare hash to prevent saving of duplicates with same name, fix for #3465
                            image_is_duplicate = True
                            break
                        filename = f"{split[0]} ({i}){split[1]}"
                        filepath = os.path.join(full_output_folder, filename)
                        i += 1

                if not image_is_duplicate:
                    if image_save_function is not None:
                        image_save_function(image, post, filepath)
                    else:
                        with open(filepath, "wb") as f:
                            f.write(image.file.read())

                return web.json_response({"name" : filename, "subfolder": subfolder, "type": image_upload_type})
            else:
                return web.Response(status=400)

        @routes.post("/upload/image")
        async def upload_image(request):
            post = await request.post()
            return image_upload(post)


        @routes.post("/upload/mask")
        async def upload_mask(request):
            post = await request.post()

            def image_save_function(image, post, filepath):
                original_ref = json.loads(post.get("original_ref"))
                filename, output_dir = folder_paths.annotated_filepath(original_ref['filename'])

                # validation for security: prevent accessing arbitrary path
                if filename[0] == '/' or '..' in filename:
                    return web.Response(status=400)

                if output_dir is None:
                    type = original_ref.get("type", "output")
                    output_dir = folder_paths.get_directory_by_type(type)

                if output_dir is None:
                    return web.Response(status=400)

                if original_ref.get("subfolder", "") != "":
                    full_output_dir = os.path.join(output_dir, original_ref["subfolder"])
                    if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                        return web.Response(status=403)
                    output_dir = full_output_dir

                file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    with Image.open(file) as original_pil:
                        metadata = PngInfo()
                        if hasattr(original_pil,'text'):
                            for key in original_pil.text:
                                metadata.add_text(key, original_pil.text[key])
                        original_pil = original_pil.convert('RGBA')
                        mask_pil = Image.open(image.file).convert('RGBA')

                        # alpha copy
                        new_alpha = mask_pil.getchannel('A')
                        original_pil.putalpha(new_alpha)
                        original_pil.save(filepath, compress_level=4, pnginfo=metadata)

            return image_upload(post, image_save_function)

        @routes.get("/view")
        async def view_image(request):
            if "filename" in request.rel_url.query:
                filename = request.rel_url.query["filename"]
                filename,output_dir = folder_paths.annotated_filepath(filename)

                # validation for security: prevent accessing arbitrary path
                if filename[0] == '/' or '..' in filename:
                    return web.Response(status=400)

                if output_dir is None:
                    type = request.rel_url.query.get("type", "output")
                    output_dir = folder_paths.get_directory_by_type(type)

                if output_dir is None:
                    return web.Response(status=400)

                if "subfolder" in request.rel_url.query:
                    full_output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])
                    if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                        return web.Response(status=403)
                    output_dir = full_output_dir

                filename = os.path.basename(filename)
                file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    if 'preview' in request.rel_url.query:
                        with Image.open(file) as img:
                            preview_info = request.rel_url.query['preview'].split(';')
                            image_format = preview_info[0]
                            if image_format not in ['webp', 'jpeg'] or 'a' in request.rel_url.query.get('channel', ''):
                                image_format = 'webp'

                            quality = 90
                            if preview_info[-1].isdigit():
                                quality = int(preview_info[-1])

                            buffer = BytesIO()
                            if image_format in ['jpeg'] or request.rel_url.query.get('channel', '') == 'rgb':
                                img = img.convert("RGB")
                            img.save(buffer, format=image_format, quality=quality)
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type=f'image/{image_format}',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    if 'channel' not in request.rel_url.query:
                        channel = 'rgba'
                    else:
                        channel = request.rel_url.query["channel"]

                    if channel == 'rgb':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                r, g, b, a = img.split()
                                new_img = Image.merge('RGB', (r, g, b))
                            else:
                                new_img = img.convert("RGB")

                            buffer = BytesIO()
                            new_img.save(buffer, format='PNG')
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    elif channel == 'a':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                _, _, _, a = img.split()
                            else:
                                a = Image.new('L', img.size, 255)

                            # alpha img
                            alpha_img = Image.new('RGBA', img.size)
                            alpha_img.putalpha(a)
                            alpha_buffer = BytesIO()
                            alpha_img.save(alpha_buffer, format='PNG')
                            alpha_buffer.seek(0)

                            return web.Response(body=alpha_buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})
                    else:
                        return web.FileResponse(file, headers={"Content-Disposition": f"filename=\"{filename}\""})

            return web.Response(status=404)

        @routes.get("/view_metadata/{folder_name}")
        async def view_metadata(request):
            folder_name = request.match_info.get("folder_name", None)
            if folder_name is None:
                return web.Response(status=404)
            if not "filename" in request.rel_url.query:
                return web.Response(status=404)

            filename = request.rel_url.query["filename"]
            if not filename.endswith(".safetensors"):
                return web.Response(status=404)

            safetensors_path = folder_paths.get_full_path(folder_name, filename)
            if safetensors_path is None:
                return web.Response(status=404)
            out = comfy.utils.safetensors_header(safetensors_path, max_size=1024*1024)
            if out is None:
                return web.Response(status=404)
            dt = json.loads(out)
            if not "__metadata__" in dt:
                return web.Response(status=404)
            return web.json_response(dt["__metadata__"])

        @routes.get("/system_stats")
        async def get_queue(request):
            device = comfy.model_management.get_torch_device()
            device_name = comfy.model_management.get_torch_device_name(device)
            vram_total, torch_vram_total = comfy.model_management.get_total_memory(device, torch_total_too=True)
            vram_free, torch_vram_free = comfy.model_management.get_free_memory(device, torch_free_too=True)
            system_stats = {
                "system": {
                    "os": os.name,
                    "python_version": sys.version,
                    "embedded_python": os.path.split(os.path.split(sys.executable)[0])[1] == "python_embeded"
                },
                "devices": [
                    {
                        "name": device_name,
                        "type": device.type,
                        "index": device.index,
                        "vram_total": vram_total,
                        "vram_free": vram_free,
                        "torch_vram_total": torch_vram_total,
                        "torch_vram_free": torch_vram_free,
                    }
                ]
            }
            return web.json_response(system_stats)

        @routes.get("/prompt")
        async def get_prompt(request):
            return web.json_response(self.get_queue_info())

        def node_info(node_class):
            obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
            info = {}
            info['input'] = obj_class.INPUT_TYPES()
            info['input_order'] = {key: list(value.keys()) for (key, value) in obj_class.INPUT_TYPES().items()}
            info['output'] = obj_class.RETURN_TYPES
            info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [False] * len(obj_class.RETURN_TYPES)
            info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
            info['name'] = node_class
            info['display_name'] = nodes.NODE_DISPLAY_NAME_MAPPINGS[node_class] if node_class in nodes.NODE_DISPLAY_NAME_MAPPINGS.keys() else node_class
            info['description'] = obj_class.DESCRIPTION if hasattr(obj_class,'DESCRIPTION') else ''
            info['python_module'] = getattr(obj_class, "RELATIVE_PYTHON_MODULE", "nodes")
            info['category'] = 'sd'
            if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE == True:
                info['output_node'] = True
            else:
                info['output_node'] = False

            if hasattr(obj_class, 'CATEGORY'):
                info['category'] = obj_class.CATEGORY

            if hasattr(obj_class, 'OUTPUT_TOOLTIPS'):
                info['output_tooltips'] = obj_class.OUTPUT_TOOLTIPS

            if getattr(obj_class, "DEPRECATED", False):
                info['deprecated'] = True
            if getattr(obj_class, "EXPERIMENTAL", False):
                info['experimental'] = True
            return info

        @routes.get("/object_info")
        async def get_object_info(request):
            out = {}
            for x in nodes.NODE_CLASS_MAPPINGS:
                try:
                    out[x] = node_info(x)
                except Exception as e:
                    logging.error(f"[ERROR] An error occurred while retrieving information for the '{x}' node.")
                    logging.error(traceback.format_exc())
            return web.json_response(out)

        @routes.get("/object_info/{node_class}")
        async def get_object_info_node(request):
            node_class = request.match_info.get("node_class", None)
            out = {}
            if (node_class is not None) and (node_class in nodes.NODE_CLASS_MAPPINGS):
                out[node_class] = node_info(node_class)
            return web.json_response(out)

        @routes.get("/history")
        async def get_history(request):
            max_items = request.rel_url.query.get("max_items", None)
            if max_items is not None:
                max_items = int(max_items)
            return web.json_response(self.prompt_queue.get_history(max_items=max_items))

        @routes.get("/history/{prompt_id}")
        async def get_history(request):
            prompt_id = request.match_info.get("prompt_id", None)
            return web.json_response(self.prompt_queue.get_history(prompt_id=prompt_id))

        @routes.get("/queue")
        async def get_queue(request):
            queue_info = {}
            current_queue = self.prompt_queue.get_current_queue()
            queue_info['queue_running'] = current_queue[0]
            queue_info['queue_pending'] = current_queue[1]
            return web.json_response(queue_info)

        @routes.post("/prompt")
        async def post_prompt(request):
            logging.info("got prompt")
            resp_code = 200
            out_string = ""
            json_data =  await request.json()
            json_data = self.trigger_on_prompt(json_data)

            if "number" in json_data:
                number = float(json_data['number'])
            else:
                number = self.number
                if "front" in json_data:
                    if json_data['front']:
                        number = -number

                self.number += 1

            if "prompt" in json_data:
                prompt = json_data["prompt"]
                valid = execution.validate_prompt(prompt)
                extra_data = {}
                if "extra_data" in json_data:
                    extra_data = json_data["extra_data"]

                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]
                if valid[0]:
                    prompt_id = str(uuid.uuid4())
                    outputs_to_execute = valid[2]
                    self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
                    response = {"prompt_id": prompt_id, "number": number, "node_errors": valid[3]}
                    return web.json_response(response)
                else:
                    logging.warning("invalid prompt: {}".format(valid[1]))
                    return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)
            else:
                return web.json_response({"error": "no prompt", "node_errors": []}, status=400)

        @routes.post("/queue")
        async def post_queue(request):
            json_data =  await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_queue()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    delete_func = lambda a: a[1] == id_to_delete
                    self.prompt_queue.delete_queue_item(delete_func)

            return web.Response(status=200)

        @routes.post("/interrupt")
        async def post_interrupt(request):
            nodes.interrupt_processing()
            return web.Response(status=200)

        @routes.post("/free")
        async def post_free(request):
            json_data = await request.json()
            unload_models = json_data.get("unload_models", False)
            free_memory = json_data.get("free_memory", False)
            if unload_models:
                self.prompt_queue.set_flag("unload_models", unload_models)
            if free_memory:
                self.prompt_queue.set_flag("free_memory", free_memory)
            return web.Response(status=200)

        @routes.post("/history")
        async def post_history(request):
            json_data =  await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_history()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    self.prompt_queue.delete_history_item(id_to_delete)

            return web.Response(status=200)
        
        # Internal route. Should not be depended upon and is subject to change at any time.
        # TODO(robinhuang): Move to internal route table class once we refactor PromptServer to pass around Websocket.
        @routes.post("/internal/models/download")
        async def download_handler(request):
            async def report_progress(filename: str, status: DownloadModelStatus):
                await self.send_json("download_progress", status.to_dict())

            data = await request.json()
            url = data.get('url')
            model_directory = data.get('model_directory')
            model_filename = data.get('model_filename')
            progress_interval = data.get('progress_interval', 1.0) # In seconds, how often to report download progress.

            if not url or not model_directory or not model_filename:
                return web.json_response({"status": "error", "message": "Missing URL or folder path or filename"}, status=400)

            session = self.client_session
            if session is None:
                logging.error("Client session is not initialized")
                return web.Response(status=500)
            
            task = asyncio.create_task(download_model(lambda url: session.get(url), model_filename, url, model_directory, report_progress, progress_interval))
            await task

            return web.json_response(task.result().to_dict())

        @routes.post("/generate_image")
        async def generate_image(request):
            try:
                logging.info("Started processing request in generate_image")

                # Parse JSON request body
                json_data = await request.json()
                prompt = json_data.get("prompt")
                if not prompt:
                    return web.json_response({"error": "prompt is required"}, status=400)

                # Process the prompt
                valid = execution.validate_prompt(prompt)
                if not valid[0]:
                    logging.warning(f"Invalid prompt: {valid[1]}")
                    return web.json_response({"error": valid[1]}, status=400)

                # Generate a unique ID for this prompt
                prompt_id = str(uuid.uuid4())
                logging.info(f"Generated prompt_id: {prompt_id} for prompt: {prompt}")

                number = self.number
                self.number += 1

                # Prepare the extra data if available
                extra_data = json_data.get("extra_data", {})
                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]

                # Add prompt to the queue
                try:
                    outputs_to_execute = valid[2]
                    self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
                except asyncio.QueueFull:
                    logging.error(f"Queue is full, cannot process prompt_id {prompt_id}")
                    return web.json_response({"error": "Server is busy, please try again later"}, status=503)
                logging.info("Prompt successfully submitted to the queue")

                # Establish a WebSocket connection manually
                sid = extra_data.get("client_id", None)
                if not sid:
                    return web.json_response({"error": "client_id is required"}, status=400)

                websocket_url = f"ws://localhost:7860/ws?clientId={sid}"
                logging.info(f"Connecting to WebSocket URL: {websocket_url}")

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(websocket_url) as ws:
                        logging.info(f"Connected to WebSocket with client_id: {sid}")

                        # Listen for WebSocket messages until the 'executed' message is received
                        image_filename = None
                        while True:
                            try:
                                msg = await ws.receive(timeout=60)  # 60 seconds timeout for receiving a message
                                logging.info(f"Received WebSocket message: {msg}")

                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    if data.get('type') == 'executed' and data['data']['prompt_id'] == prompt_id:
                                        output = data['data']['output']
                                        if 'images' in output and output['images']:
                                            image_info = output['images'][0]

                                            logging.info(f"Image info: {image_info}")

                                            image_filename = image_info.get('filename')
                                            break
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    logging.warning(f"WebSocket error: {ws.exception()}")

                            except asyncio.TimeoutError:
                                logging.warning(f"Timeout while waiting for 'executed' message for prompt_id {prompt_id}")
                                return web.json_response({"error": "Image generation timed out"}, status=500)

                        if image_filename:
                            # Construct the correct path to the image using absolute path
                            output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output'))
                            path = os.path.join(output_dir, image_filename)

                            try:
                                # Upload the image to AWS S3
                                s3_client = boto3.client(
                                    's3',
                                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                    region_name=AWS_REGION,
                                )
                                new_filename = str(uuid.uuid4()) + '.jpg'
                                s3_key = f"devEnv/{new_filename}" 

                                s3_client.upload_file(path, AWS_BUCKET_NAME, s3_key,ExtraArgs={'ACL': 'public-read'})

                                # Get the S3 URL
                                s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com.cn/{s3_key}"

                                logging.info("Image successfully uploaded to S3")
                                return web.json_response({"image_url": s3_url})
                            except NoCredentialsError:
                                logging.error("AWS credentials not available")
                                return web.json_response({"error": "AWS credentials not available"}, status=500)
                            except ClientError as e:
                                logging.error(f"Failed to upload image to S3: {str(e)}")
                                return web.json_response({"error": "Failed to upload image to S3"}, status=500)
                            except Exception as e:
                                logging.error(f"Failed to process image: {str(e)}")
                                return web.json_response({"error": "Failed to process the image"}, status=500)
                        else:
                            return web.json_response({"error": "Failed to generate image"}, status=500)

            except Exception as e:
                logging.error(f"Error in generate_image for prompt_id {prompt_id}: {str(e)}")
                logging.error(traceback.format_exc())
                return web.json_response({"error": "Internal server error"}, status=500)


        @routes.post("/generate_image_01_idea")
        async def generate_image_01_idea(request):
            try:
                logging.info("Started processing request in generate_image")

                # Parse JSON request body
                json_data = await request.json()
                prompt = json_data.get("prompt")
                if not prompt:
                    return web.json_response({"error": "prompt is required"}, status=400)

                # Extract and validate the airi_path parameter
                airi_path = json_data.get("airi_path")
                if not airi_path or airi_path in ["", "null", "undefined", None]:
                    # Replace the image name with "image.png" if airi_path is invalid
                    if '30' in prompt:
                        prompt['30']['inputs']['image'] = "image.png"
                    airi_path = None  # Explicitly set to None for clarity
                else:
                    # Generate a GUID for the image filename only if airi_path is valid
                    image_filename = str(uuid.uuid4()) + os.path.splitext(airi_path)[1]
                    input_dir = os.path.abspath(os.path.join(os.getcwd(), 'input'))
                    local_image_path = os.path.join(input_dir, image_filename)

                    # Download the image from the S3 path and save it to the input directory
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(airi_path) as resp:
                                if resp.status == 200:
                                    with open(local_image_path, 'wb') as f:
                                        f.write(await resp.read())
                                    logging.info(f"Image successfully downloaded and saved to {local_image_path}")
                                else:
                                    logging.error(f"Failed to download image, status code: {resp.status}")
                                    return web.json_response({"error": "Failed to download image"}, status=500)
                    except Exception as e:
                        logging.error(f"Failed to download image from S3: {str(e)}")
                        return web.json_response({"error": "Failed to download image from S3"}, status=500)

                    # Update the payload with the new image path (GUID)
                    prompt['30']['inputs']['image'] = image_filename

                # Process the prompt
                valid = execution.validate_prompt(prompt)
                if not valid[0]:
                    logging.warning(f"Invalid prompt: {valid[1]}")
                    return web.json_response({"error": valid[1]}, status=400)

                # Generate a unique ID for this prompt
                prompt_id = str(uuid.uuid4())

                number = self.number
                self.number += 1

                # Prepare the extra data if available
                extra_data = json_data.get("extra_data", {})
                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]

                # Add prompt to the queue
                try:
                    outputs_to_execute = valid[2]
                    self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
                except asyncio.QueueFull:
                    logging.error(f"Queue is full, cannot process prompt_id {prompt_id}")
                    return web.json_response({"error": "Server is busy, please try again later"}, status=503)
                logging.info("Prompt successfully submitted to the queue")

                # Establish a WebSocket connection manually
                sid = extra_data.get("client_id", None)
                if not sid:
                    return web.json_response({"error": "client_id is required"}, status=400)

                websocket_url = f"ws://localhost:7860/ws?clientId={sid}"
                logging.info(f"Connecting to WebSocket URL: {websocket_url}")

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(websocket_url) as ws:
                        logging.info(f"Connected to WebSocket with client_id: {sid}")

                        image_filenames = []
                        while True:
                            try:
                                msg = await ws.receive(timeout=600)  # 10 minutes timeout for receiving a message
                                logging.info(f"Received WebSocket message: {msg.data}")

                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    logging.info(f"WebSocket message content: {json.dumps(data, indent=2)}")

                                    if data.get('type') == 'executed' and data['data'].get('prompt_id') == prompt_id:
                                        output = data['data']['output']
                                        if 'images' in output:
                                            if output['images']:  # If images list is not empty
                                                for image_info in output['images']:
                                                    logging.info(f"Image info: {image_info}")
                                                    image_filenames.append(image_info.get('filename'))
                                                break
                                            else:
                                                logging.info(f"Received 'executed' message but images are empty. Waiting for images...")
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    logging.warning(f"WebSocket error: {ws.exception()}")

                            except asyncio.TimeoutError:
                                logging.warning(f"Timeout while waiting for 'executed' message with images for prompt_id {prompt_id}")
                                return web.json_response({"error": "Image generation timed out"}, status=500)

                        if image_filenames:
                            s3_urls = []
                            thumbnail_urls = []
                            image_metadata = []
                            output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output'))

                            for image_filename in image_filenames:
                                path = os.path.join(output_dir, image_filename)

                                try:
                                    # Open the original image
                                    image = Image.open(path)

                                    # Get image size and dimensions
                                    width, height = image.size
                                    file_size = os.path.getsize(path)

                                    # Store the image metadata
                                    image_metadata.append({
                                        "filename": image_filename,
                                        "width": width,
                                        "height": height,
                                        "size_in_bytes": file_size
                                    })

                                    # Create a thumbnail
                                    image.thumbnail((200, 200))

                                    # Save the thumbnail
                                    thumbnail_filename = "thumbnail_" + image_filename
                                    thumbnail_path = os.path.join(output_dir, thumbnail_filename)
                                    image.save(thumbnail_path)

                                    # Upload both original and thumbnail to S3
                                    s3_client = boto3.client(
                                        's3',
                                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                        region_name=AWS_REGION,
                                    )
                                    new_filename = str(uuid.uuid4()) + '.jpg'
                                    thumbnail_s3_key = f"devEnv/thumbnail_{new_filename}"
                                    original_s3_key = f"devEnv/{new_filename}"

                                    # Upload the original image
                                    s3_client.upload_file(path, AWS_BUCKET_NAME, original_s3_key, ExtraArgs={'ACL': 'public-read'})
                                    s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com.cn/{original_s3_key}"
                                    s3_urls.append(s3_url)

                                    # Upload the thumbnail image
                                    s3_client.upload_file(thumbnail_path, AWS_BUCKET_NAME, thumbnail_s3_key, ExtraArgs={'ACL': 'public-read'})
                                    thumbnail_s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com.cn/{thumbnail_s3_key}"
                                    thumbnail_urls.append(thumbnail_s3_url)

                                    logging.info("Images and thumbnails successfully uploaded to S3")

                                except NoCredentialsError:
                                    logging.error("AWS credentials not available")
                                    return web.json_response({"error": "AWS credentials not available"}, status=500)
                                except ClientError as e:
                                    logging.error(f"Failed to upload image to S3: {str(e)}")
                                    return web.json_response({"error": "Failed to upload image to S3"}, status=500)
                                except Exception as e:
                                    logging.error(f"Failed to process image: {str(e)}")
                                    return web.json_response({"error": "Failed to process the image"}, status=500)

                            return web.json_response({
                                "image_urls": s3_urls, 
                                "image_thumbnail_urls": thumbnail_urls, 
                                "image_metadata": image_metadata
                            })
                        else:
                            logging.error(f"No images found in the output for prompt_id {prompt_id}")
                            return web.json_response({"error": "Failed to generate image"}, status=500)

            except Exception as e:
                logging.error(f"Error in generate_image: {str(e)}")
                logging.error(traceback.format_exc())
                return web.json_response({"error": "Internal server error"}, status=500)


        @routes.post("/generate_image_03_render")
        async def generate_image_03_render(request):
            try:
                logging.info("Started processing request in generate_image")

                # Parse JSON request body
                json_data = await request.json()
                prompt = json_data.get("prompt")
                if not prompt:
                    return web.json_response({"error": "prompt is required"}, status=400)

                # Extract and validate the airi_path parameter
                airi_path = json_data.get("airi_path")
                if not airi_path or airi_path in ["", "null", "undefined", None]:
                    # Replace the image name with "sketch.jpg" if airi_path is invalid
                    if '30' in prompt:
                        prompt['30']['inputs']['image'] = "image.png"
                    airi_path = None  # Explicitly set to None for clarity
                else:
                    # Generate a GUID for the image filename only if airi_path is valid
                    image_filename = str(uuid.uuid4()) + os.path.splitext(airi_path)[1]
                    input_dir = os.path.abspath(os.path.join(os.getcwd(), 'input'))
                    local_image_path = os.path.join(input_dir, image_filename)

                    # Download the image from the S3 path and save it to the input directory
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(airi_path) as resp:
                                if resp.status == 200:
                                    with open(local_image_path, 'wb') as f:
                                        f.write(await resp.read())
                                    logging.info(f"Image successfully downloaded and saved to {local_image_path}")
                                else:
                                    logging.error(f"Failed to download image, status code: {resp.status}")
                                    return web.json_response({"error": "Failed to download image"}, status=500)
                    except Exception as e:
                        logging.error(f"Failed to download image from S3: {str(e)}")
                        return web.json_response({"error": "Failed to download image from S3"}, status=500)

                    # Update the payload with the new image path (GUID)
                    prompt['30']['inputs']['image'] = image_filename

                # Extract and handle the base_image parameter (similar to airi_path)
                base_image = json_data.get("base_image")
                if base_image and base_image not in ["", "null", "undefined", None]:
                    # Generate a GUID for the base image filename
                    base_image_filename = str(uuid.uuid4()) + os.path.splitext(base_image)[1]
                    base_image_dir = os.path.abspath(os.path.join(os.getcwd(), 'input'))
                    local_base_image_path = os.path.join(base_image_dir, base_image_filename)

                    # Download the base image from the S3 path and save it to the input directory
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(base_image) as resp:
                                if resp.status == 200:
                                    with open(local_base_image_path, 'wb') as f:
                                        f.write(await resp.read())
                                    logging.info(f"Base image successfully downloaded and saved to {local_base_image_path}")
                                else:
                                    logging.error(f"Failed to download base image, status code: {resp.status}")
                                    return web.json_response({"error": "Failed to download base image"}, status=500)
                    except Exception as e:
                        logging.error(f"Failed to download base image from S3: {str(e)}")
                        return web.json_response({"error": "Failed to download base image from S3"}, status=500)

                    # Update the prompt to include the base image in section '35'
                    if '35' in prompt:
                        prompt['35']['inputs']['image'] = base_image_filename
                    else:
                        # If '35' does not exist, create it and add the base image
                        prompt['35'] = {
                            "inputs": {
                                "image": base_image_filename
                            }
                        }

                # Process the prompt
                valid = execution.validate_prompt(prompt)
                if not valid[0]:
                    logging.warning(f"Invalid prompt: {valid[1]}")
                    return web.json_response({"error": valid[1]}, status=400)

                # Generate a unique ID for this prompt
                prompt_id = str(uuid.uuid4())

                number = self.number
                self.number += 1

                # Prepare the extra data if available
                extra_data = json_data.get("extra_data", {})
                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]

                # Add prompt to the queue
                try:
                    outputs_to_execute = valid[2]
                    self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
                except asyncio.QueueFull:
                    logging.error(f"Queue is full, cannot process prompt_id {prompt_id}")
                    return web.json_response({"error": "Server is busy, please try again later"}, status=503)
                logging.info("Prompt successfully submitted to the queue")

                # Establish a WebSocket connection manually
                sid = extra_data.get("client_id", None)
                if not sid:
                    return web.json_response({"error": "client_id is required"}, status=400)

                websocket_url = f"ws://localhost:7860/ws?clientId={sid}"
                logging.info(f"Connecting to WebSocket URL: {websocket_url}")

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(websocket_url) as ws:
                        logging.info(f"Connected to WebSocket with client_id: {sid}")

                        image_filenames = []
                        while True:
                            try:
                                msg = await ws.receive(timeout=600)  # 10 minutes timeout for receiving a message
                                logging.info(f"Received WebSocket message: {msg.data}")

                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    logging.info(f"WebSocket message content: {json.dumps(data, indent=2)}")

                                    if data.get('type') == 'executed' and data['data'].get('prompt_id') == prompt_id:
                                        output = data['data']['output']
                                        if 'images' in output:
                                            if output['images']:  # If images list is not empty
                                                for image_info in output['images']:
                                                    logging.info(f"Image info: {image_info}")
                                                    image_filenames.append(image_info.get('filename'))
                                                break
                                            else:
                                                logging.info(f"Received 'executed' message but images are empty. Waiting for images...")
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    logging.warning(f"WebSocket error: {ws.exception()}")

                            except asyncio.TimeoutError:
                                logging.warning(f"Timeout while waiting for 'executed' message with images for prompt_id {prompt_id}")
                                return web.json_response({"error": "Image generation timed out"}, status=500)

                        if image_filenames:
                            s3_urls = []
                            thumbnail_urls = []
                            image_metadata = []
                            output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output'))
                            for image_filename in image_filenames:
                                path = os.path.join(output_dir, image_filename)

                                # Resize the obtained image to create a thumbnail using Pillow
                                try:
                                    # Open the original image
                                    image = Image.open(path)

                                    # Get image size and dimensions
                                    width, height = image.size
                                    file_size = os.path.getsize(path)

                                    # Store the image metadata
                                    image_metadata.append({
                                        "filename": image_filename,
                                        "width": width,
                                        "height": height,
                                        "size_in_bytes": file_size
                                    })

                                    # Create a thumbnail
                                    image.thumbnail((200, 200))

                                    # Save the thumbnail
                                    thumbnail_filename = "thumbnail_" + image_filename
                                    thumbnail_path = os.path.join(output_dir, thumbnail_filename)
                                    image.save(thumbnail_path)

                                    # Upload both original and thumbnail to S3
                                    s3_client = boto3.client(
                                        's3',
                                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                        region_name=AWS_REGION,
                                    )
                                    new_filename = str(uuid.uuid4()) + '.jpg'
                                    thumbnail_s3_key = f"devEnv/thumbnail_{new_filename}"
                                    original_s3_key = f"devEnv/{new_filename}"

                                    # Upload the original image
                                    s3_client.upload_file(path, AWS_BUCKET_NAME, original_s3_key, ExtraArgs={'ACL': 'public-read'})
                                    s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com.cn/{original_s3_key}"
                                    s3_urls.append(s3_url)

                                    # Upload the thumbnail image
                                    s3_client.upload_file(thumbnail_path, AWS_BUCKET_NAME, thumbnail_s3_key, ExtraArgs={'ACL': 'public-read'})
                                    thumbnail_s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com.cn/{thumbnail_s3_key}"
                                    thumbnail_urls.append(thumbnail_s3_url)

                                    logging.info("Images and thumbnails successfully uploaded to S3")

                                except NoCredentialsError:
                                    logging.error("AWS credentials not available")
                                    return web.json_response({"error": "AWS credentials not available"}, status=500)
                                except ClientError as e:
                                    logging.error(f"Failed to upload image to S3: {str(e)}")
                                    return web.json_response({"error": "Failed to upload image to S3"}, status=500)
                                except Exception as e:
                                    logging.error(f"Failed to process image: {str(e)}")
                                    return web.json_response({"error": "Failed to process the image"}, status=500)

                            return web.json_response({
                                "image_urls": s3_urls, 
                                "image_thumbnail_urls": thumbnail_urls, 
                                "image_metadata": image_metadata
                            })
                        else:
                            logging.error(f"No images found in the output for prompt_id {prompt_id}")
                            return web.json_response({"error": "Failed to generate image"}, status=500)

            except Exception as e:
                logging.error(f"Error in generate_image: {str(e)}")
                logging.error(traceback.format_exc())
                return web.json_response({"error": "Internal server error"}, status=500)


        # custom api for the edit workflow airi
        @routes.post("/generate_image_03_edit")
        async def generate_image_03_edit(request):
            try:
                logging.info("Started processing request in generate_image")

                # Parse JSON request body
                json_data = await request.json()
                prompt = json_data.get("prompt")
                if not prompt:
                    return web.json_response({"error": "prompt is required"}, status=400)

                # Extract and validate the airi_path parameter
                airi_path = json_data.get("airi_path")
                if not airi_path or airi_path in ["", "null", "undefined", None]:
                    # Replace the image name with "image.png" if airi_path is invalid
                    if '30' in prompt:
                        prompt['30']['inputs']['image'] = "image.png"
                    airi_path = None  # Explicitly set to None for clarity
                else:
                    # Generate a GUID for the image filename only if airi_path is valid
                    image_filename = str(uuid.uuid4()) + os.path.splitext(airi_path)[1]
                    input_dir = os.path.abspath(os.path.join(os.getcwd(), 'input'))
                    local_image_path = os.path.join(input_dir, image_filename)

                    # Download the image from the S3 path and save it to the input directory
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(airi_path) as resp:
                                if resp.status == 200:
                                    with open(local_image_path, 'wb') as f:
                                        f.write(await resp.read())
                                    logging.info(f"Image successfully downloaded and saved to {local_image_path}")
                                else:
                                    logging.error(f"Failed to download image, status code: {resp.status}")
                                    return web.json_response({"error": "Failed to download image"}, status=500)
                    except Exception as e:
                        logging.error(f"Failed to download image from S3: {str(e)}")
                        return web.json_response({"error": "Failed to download image from S3"}, status=500)

                    # Update the payload with the new image path (GUID)
                    prompt['30']['inputs']['image'] = image_filename

                # Extract and handle the base_image parameter (similar to airi_path)
                base_image = json_data.get("base_image")
                if base_image and base_image not in ["", "null", "undefined", None]:
                    # Generate a GUID for the base image filename
                    base_image_filename = str(uuid.uuid4()) + os.path.splitext(base_image)[1]
                    base_image_dir = os.path.abspath(os.path.join(os.getcwd(), 'input'))
                    local_base_image_path = os.path.join(base_image_dir, base_image_filename)

                    # Download the base image from the S3 path and save it to the input directory
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(base_image) as resp:
                                if resp.status == 200:
                                    with open(local_base_image_path, 'wb') as f:
                                        f.write(await resp.read())
                                    logging.info(f"Base image successfully downloaded and saved to {local_base_image_path}")
                                else:
                                    logging.error(f"Failed to download base image, status code: {resp.status}")
                                    return web.json_response({"error": "Failed to download base image"}, status=500)
                    except Exception as e:
                        logging.error(f"Failed to download base image from S3: {str(e)}")
                        return web.json_response({"error": "Failed to download base image from S3"}, status=500)

                    # Update the prompt to include the base image in section '35'
                    if '35' in prompt:
                        prompt['35']['inputs']['image'] = base_image_filename
                    else:
                        # If '35' does not exist, create it and add the base image
                        prompt['35'] = {
                            "inputs": {
                                "image": base_image_filename
                            }
                        }

                # Extract and handle the mask_image parameter (similar to base_image)
                mask_image = json_data.get("mask_image")
                if mask_image and mask_image not in ["", "null", "undefined", None]:
                    mask_image_filename = str(uuid.uuid4()) + os.path.splitext(mask_image)[1]
                    mask_image_dir = os.path.abspath(os.path.join(os.getcwd(), 'input'))
                    local_mask_image_path = os.path.join(mask_image_dir, mask_image_filename)

                    # Download the mask image from the S3 path and save it to the input directory
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(mask_image) as resp:
                                if resp.status == 200:
                                    with open(local_mask_image_path, 'wb') as f:
                                        f.write(await resp.read())
                                    logging.info(f"Mask image successfully downloaded and saved to {local_mask_image_path}")
                                else:
                                    logging.error(f"Failed to download mask image, status code: {resp.status}")
                                    return web.json_response({"error": "Failed to download mask image"}, status=500)
                    except Exception as e:
                        logging.error(f"Failed to download mask image from S3: {str(e)}")
                        return web.json_response({"error": "Failed to download mask image from S3"}, status=500)

                    # Update the prompt to include the mask image in section '101'
                    if '101' in prompt:
                        prompt['101']['inputs']['image'] = mask_image_filename
                    else:
                        prompt['101'] = {
                            "inputs": {
                                "image": mask_image_filename
                            }
                        }

                # Process the prompt
                valid = execution.validate_prompt(prompt)
                if not valid[0]:
                    logging.warning(f"Invalid prompt: {valid[1]}")
                    return web.json_response({"error": valid[1]}, status=400)

                # Generate a unique ID for this prompt
                prompt_id = str(uuid.uuid4())

                number = self.number
                self.number += 1

                # Prepare the extra data if available
                extra_data = json_data.get("extra_data", {})
                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]

                # Add prompt to the queue
                try:
                    outputs_to_execute = valid[2]
                    self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
                except asyncio.QueueFull:
                    logging.error(f"Queue is full, cannot process prompt_id {prompt_id}")
                    return web.json_response({"error": "Server is busy, please try again later"}, status=503)
                logging.info("Prompt successfully submitted to the queue")

                # Establish a WebSocket connection manually
                sid = extra_data.get("client_id", None)
                if not sid:
                    return web.json_response({"error": "client_id is required"}, status=400)

                websocket_url = f"ws://localhost:7860/ws?clientId={sid}"
                logging.info(f"Connecting to WebSocket URL: {websocket_url}")

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(websocket_url) as ws:
                        logging.info(f"Connected to WebSocket with client_id: {sid}")

                        image_filenames = []
                        while True:
                            try:
                                msg = await ws.receive(timeout=600)  # 10 minutes timeout for receiving a message
                                logging.info(f"Received WebSocket message: {msg.data}")

                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    logging.info(f"WebSocket message content: {json.dumps(data, indent=2)}")

                                    if data.get('type') == 'executed' and data['data'].get('prompt_id') == prompt_id:
                                        output = data['data']['output']
                                        if 'images' in output:
                                            if output['images']:  # If images list is not empty
                                                for image_info in output['images']:
                                                    logging.info(f"Image info: {image_info}")
                                                    image_filenames.append(image_info.get('filename'))
                                                break
                                            else:
                                                logging.info(f"Received 'executed' message but images are empty. Waiting for images...")
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    logging.warning(f"WebSocket error: {ws.exception()}")

                            except asyncio.TimeoutError:
                                logging.warning(f"Timeout while waiting for 'executed' message with images for prompt_id {prompt_id}")
                                return web.json_response({"error": "Image generation timed out"}, status=500)

                        if image_filenames:
                            s3_urls = []
                            thumbnail_urls = []
                            image_metadata = []
                            output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output'))
                            for image_filename in image_filenames:
                                path = os.path.join(output_dir, image_filename)

                                # Resize the obtained image to create a thumbnail using Pillow
                                try:
                                    # Open the original image
                                    image = Image.open(path)

                                    # Capture image metadata
                                    width, height = image.size
                                    size_in_bytes = os.path.getsize(path)
                                    image_metadata.append({
                                        "filename": image_filename,
                                        "width": width,
                                        "height": height,
                                        "size_in_bytes": size_in_bytes
                                    })

                                    # Create a thumbnail
                                    image.thumbnail((200, 200))

                                    # Save the thumbnail
                                    thumbnail_filename = "thumbnail_" + image_filename
                                    thumbnail_path = os.path.join(output_dir, thumbnail_filename)
                                    image.save(thumbnail_path)

                                    # Upload both original and thumbnail to S3
                                    s3_client = boto3.client(
                                        's3',
                                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                        region_name=AWS_REGION,
                                    )
                                    new_filename = str(uuid.uuid4()) + '.jpg'
                                    thumbnail_s3_key = f"devEnv/thumbnail_{new_filename}"
                                    original_s3_key = f"devEnv/{new_filename}"

                                    # Upload the original image
                                    s3_client.upload_file(path, AWS_BUCKET_NAME, original_s3_key, ExtraArgs={'ACL': 'public-read'})
                                    s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com.cn/{original_s3_key}"
                                    s3_urls.append(s3_url)

                                    # Upload the thumbnail image
                                    s3_client.upload_file(thumbnail_path, AWS_BUCKET_NAME, thumbnail_s3_key, ExtraArgs={'ACL': 'public-read'})
                                    thumbnail_s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com.cn/{thumbnail_s3_key}"
                                    thumbnail_urls.append(thumbnail_s3_url)

                                    logging.info("Images and thumbnails successfully uploaded to S3")

                                except NoCredentialsError:
                                    logging.error("AWS credentials not available")
                                    return web.json_response({"error": "AWS credentials not available"}, status=500)
                                except ClientError as e:
                                    logging.error(f"Failed to upload image to S3: {str(e)}")
                                    return web.json_response({"error": "Failed to upload image to S3"}, status=500)
                                except Exception as e:
                                    logging.error(f"Failed to process image: {str(e)}")
                                    return web.json_response({"error": "Failed to process the image"}, status=500)

                            return web.json_response({
                                "image_urls": s3_urls,
                                "image_thumbnail_urls": thumbnail_urls,
                                "image_metadata": image_metadata
                            })
                        else:
                            logging.error(f"No images found in the output for prompt_id {prompt_id}")
                            return web.json_response({"error": "Failed to generate image"}, status=500)

            except Exception as e:
                logging.error(f"Error in generate_image: {str(e)}")
                logging.error(traceback.format_exc())
                return web.json_response({"error": "Internal server error"}, status=500)


        # custom api for the render workflow airi
        @routes.post("/generate_image_03_edit_cn_inpainting")
        async def generate_image_03_edit_cn_inpainting(request):
            try:
                logging.info("Started processing request in generate_image")

                # Parse JSON request body
                json_data = await request.json()
                prompt = json_data.get("prompt")
                if not prompt:
                    return web.json_response({"error": "prompt is required"}, status=400)

                # Extract and validate the airi_path parameter
                airi_path = json_data.get("airi_path")
                if not airi_path or airi_path in ["", "null", "undefined", None]:
                    # Replace the image name with "image.png" if airi_path is invalid
                    if '30' in prompt:
                        prompt['30']['inputs']['image'] = "image.png"
                    airi_path = None  # Explicitly set to None for clarity
                else:
                    # Generate a GUID for the image filename only if airi_path is valid
                    image_filename = str(uuid.uuid4()) + os.path.splitext(airi_path)[1]
                    input_dir = os.path.abspath(os.path.join(os.getcwd(), 'input'))
                    local_image_path = os.path.join(input_dir, image_filename)

                    # Download the image from the S3 path and save it to the input directory
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(airi_path) as resp:
                                if resp.status == 200:
                                    with open(local_image_path, 'wb') as f:
                                        f.write(await resp.read())
                                    logging.info(f"Image successfully downloaded and saved to {local_image_path}")
                                else:
                                    logging.error(f"Failed to download image, status code: {resp.status}")
                                    return web.json_response({"error": "Failed to download image"}, status=500)
                    except Exception as e:
                        logging.error(f"Failed to download image from S3: {str(e)}")
                        return web.json_response({"error": "Failed to download image from S3"}, status=500)

                    # Update the payload with the new image path (GUID)
                    prompt['30']['inputs']['image'] = image_filename

                # Extract and handle the base_image parameter (similar to airi_path)
                base_image = json_data.get("base_image")
                if base_image and base_image not in ["", "null", "undefined", None]:
                    # Generate a GUID for the base image filename
                    base_image_filename = str(uuid.uuid4()) + os.path.splitext(base_image)[1]
                    base_image_dir = os.path.abspath(os.path.join(os.getcwd(), 'input'))
                    local_base_image_path = os.path.join(base_image_dir, base_image_filename)

                    # Download the base image from the S3 path and save it to the input directory
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(base_image) as resp:
                                if resp.status == 200:
                                    with open(local_base_image_path, 'wb') as f:
                                        f.write(await resp.read())
                                    logging.info(f"Base image successfully downloaded and saved to {local_base_image_path}")
                                else:
                                    logging.error(f"Failed to download base image, status code: {resp.status}")
                                    return web.json_response({"error": "Failed to download base image"}, status=500)
                    except Exception as e:
                        logging.error(f"Failed to download base image from S3: {str(e)}")
                        return web.json_response({"error": "Failed to download base image from S3"}, status=500)

                    # Update the prompt to include the base image in section '35'
                    if '35' in prompt:
                        prompt['35']['inputs']['image'] = base_image_filename
                    else:
                        # If '35' does not exist, create it and add the base image
                        prompt['35'] = {
                            "inputs": {
                                "image": base_image_filename
                            }
                        }

                # Extract and handle the mask_image parameter (similar to base_image)
                mask_image = json_data.get("mask_image")
                if mask_image and mask_image not in ["", "null", "undefined", None]:
                    mask_image_filename = str(uuid.uuid4()) + os.path.splitext(mask_image)[1]
                    mask_image_dir = os.path.abspath(os.path.join(os.getcwd(), 'input'))
                    local_mask_image_path = os.path.join(mask_image_dir, mask_image_filename)

                    # Download the mask image from the S3 path and save it to the input directory
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(mask_image) as resp:
                                if resp.status == 200:
                                    with open(local_mask_image_path, 'wb') as f:
                                        f.write(await resp.read())
                                    logging.info(f"Mask image successfully downloaded and saved to {local_mask_image_path}")
                                else:
                                    logging.error(f"Failed to download mask image, status code: {resp.status}")
                                    return web.json_response({"error": "Failed to download mask image"}, status=500)
                    except Exception as e:
                        logging.error(f"Failed to download mask image from S3: {str(e)}")
                        return web.json_response({"error": "Failed to download mask image from S3"}, status=500)

                    # Update the prompt to include the mask image in section '101'
                    if '101' in prompt:
                        prompt['101']['inputs']['image'] = mask_image_filename
                    else:
                        prompt['101'] = {
                            "inputs": {
                                "image": mask_image_filename
                            }
                        }

                # Extract and handle the cn_image parameter (similar to base_image)
                cn_image = json_data.get("cn_image")
                if cn_image and cn_image not in ["", "null", "undefined", None]:
                    cn_image_filename = str(uuid.uuid4()) + os.path.splitext(cn_image)[1]
                    cn_image_dir = os.path.abspath(os.path.join(os.getcwd(), 'input'))
                    local_cn_image_path = os.path.join(cn_image_dir, cn_image_filename)

                    # Download the cn image from the S3 path and save it to the input directory
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(cn_image) as resp:
                                if resp.status == 200:
                                    with open(local_cn_image_path, 'wb') as f:
                                        f.write(await resp.read())
                                    logging.info(f"Mask image successfully downloaded and saved to {local_cn_image_path}")
                                else:
                                    logging.error(f"Failed to download mask image, status code: {resp.status}")
                                    return web.json_response({"error": "Failed to download mask image"}, status=500)
                    except Exception as e:
                        logging.error(f"Failed to download mask image from S3: {str(e)}")
                        return web.json_response({"error": "Failed to download mask image from S3"}, status=500)

                    # Update the prompt to include the mask image in section '101'
                    if '100' in prompt:
                        prompt['100']['inputs']['image'] = cn_image_filename
                    else:
                        prompt['100'] = {
                            "inputs": {
                                "image": cn_image_filename
                            }
                        }


                # Process the prompt
                valid = execution.validate_prompt(prompt)
                if not valid[0]:
                    logging.warning(f"Invalid prompt: {valid[1]}")
                    return web.json_response({"error": valid[1]}, status=400)

                # Generate a unique ID for this prompt
                prompt_id = str(uuid.uuid4())

                number = self.number
                self.number += 1

                # Prepare the extra data if available
                extra_data = json_data.get("extra_data", {})
                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]

                # Add prompt to the queue
                try:
                    outputs_to_execute = valid[2]
                    self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
                except asyncio.QueueFull:
                    logging.error(f"Queue is full, cannot process prompt_id {prompt_id}")
                    return web.json_response({"error": "Server is busy, please try again later"}, status=503)
                logging.info("Prompt successfully submitted to the queue")

                # Establish a WebSocket connection manually
                sid = extra_data.get("client_id", None)
                if not sid:
                    return web.json_response({"error": "client_id is required"}, status=400)

                websocket_url = f"ws://localhost:7860/ws?clientId={sid}"
                logging.info(f"Connecting to WebSocket URL: {websocket_url}")

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(websocket_url) as ws:
                        logging.info(f"Connected to WebSocket with client_id: {sid}")

                        image_filenames = []
                        while True:
                            try:
                                msg = await ws.receive(timeout=600)  # 10 minutes timeout for receiving a message
                                logging.info(f"Received WebSocket message: {msg.data}")

                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    logging.info(f"WebSocket message content: {json.dumps(data, indent=2)}")

                                    if data.get('type') == 'executed' and data['data'].get('prompt_id') == prompt_id:
                                        output = data['data']['output']
                                        if 'images' in output:
                                            if output['images']:  # If images list is not empty
                                                for image_info in output['images']:
                                                    logging.info(f"Image info: {image_info}")
                                                    image_filenames.append(image_info.get('filename'))
                                                break
                                            else:
                                                logging.info(f"Received 'executed' message but images are empty. Waiting for images...")
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    logging.warning(f"WebSocket error: {ws.exception()}")

                            except asyncio.TimeoutError:
                                logging.warning(f"Timeout while waiting for 'executed' message with images for prompt_id {prompt_id}")
                                return web.json_response({"error": "Image generation timed out"}, status=500)

                        if image_filenames:
                            s3_urls = []
                            thumbnail_urls = []
                            image_metadata = []
                            output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output'))
                            for image_filename in image_filenames:
                                path = os.path.join(output_dir, image_filename)

                                # Resize the obtained image to create a thumbnail using Pillow
                                try:
                                    # Open the original image
                                    image = Image.open(path)

                                    # Capture image metadata
                                    width, height = image.size
                                    size_in_bytes = os.path.getsize(path)
                                    image_metadata.append({
                                        "filename": image_filename,
                                        "width": width,
                                        "height": height,
                                        "size_in_bytes": size_in_bytes
                                    })

                                    # Create a thumbnail
                                    image.thumbnail((200, 200))

                                    # Save the thumbnail
                                    thumbnail_filename = "thumbnail_" + image_filename
                                    thumbnail_path = os.path.join(output_dir, thumbnail_filename)
                                    image.save(thumbnail_path)

                                    # Upload both original and thumbnail to S3
                                    s3_client = boto3.client(
                                        's3',
                                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                        region_name=AWS_REGION,
                                    )
                                    new_filename = str(uuid.uuid4()) + '.jpg'
                                    thumbnail_s3_key = f"devEnv/thumbnail_{new_filename}"
                                    original_s3_key = f"devEnv/{new_filename}"

                                    # Upload the original image
                                    s3_client.upload_file(path, AWS_BUCKET_NAME, original_s3_key, ExtraArgs={'ACL': 'public-read'})
                                    s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com.cn/{original_s3_key}"
                                    s3_urls.append(s3_url)

                                    # Upload the thumbnail image
                                    s3_client.upload_file(thumbnail_path, AWS_BUCKET_NAME, thumbnail_s3_key, ExtraArgs={'ACL': 'public-read'})
                                    thumbnail_s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com.cn/{thumbnail_s3_key}"
                                    thumbnail_urls.append(thumbnail_s3_url)

                                    logging.info("Images and thumbnails successfully uploaded to S3")

                                except NoCredentialsError:
                                    logging.error("AWS credentials not available")
                                    return web.json_response({"error": "AWS credentials not available"}, status=500)
                                except ClientError as e:
                                    logging.error(f"Failed to upload image to S3: {str(e)}")
                                    return web.json_response({"error": "Failed to upload image to S3"}, status=500)
                                except Exception as e:
                                    logging.error(f"Failed to process image: {str(e)}")
                                    return web.json_response({"error": "Failed to process the image"}, status=500)

                            return web.json_response({
                                "image_urls": s3_urls,
                                "image_thumbnail_urls": thumbnail_urls,
                                "image_metadata": image_metadata
                            })
                        else:
                            logging.error(f"No images found in the output for prompt_id {prompt_id}")
                            return web.json_response({"error": "Failed to generate image"}, status=500)

            except Exception as e:
                logging.error(f"Error in generate_image: {str(e)}")
                logging.error(traceback.format_exc())
                return web.json_response({"error": "Internal server error"}, status=500)



        @routes.post("/generate_image_04_enhance_upscale_basic_api")
        async def generate_image_04_enhance_upscale_basic_api(request):
            try:
                logging.info("Started processing request in generate_image")

                # Parse JSON request body
                json_data = await request.json()
                prompt = json_data.get("prompt")
                if not prompt:
                    return web.json_response({"error": "prompt is required"}, status=400)

                # Extract and validate the airi_path parameter
                airi_path = json_data.get("airi_path")
                if not airi_path or airi_path in ["", "null", "undefined", None]:
                    if '35' in prompt:
                        prompt['35']['inputs']['image'] = "image.png"
                    airi_path = None  # Explicitly set to None for clarity
                else:
                    image_filename = str(uuid.uuid4()) + os.path.splitext(airi_path)[1]
                    input_dir = os.path.abspath(os.path.join(os.getcwd(), 'input'))
                    local_image_path = os.path.join(input_dir, image_filename)

                    # Download the image from the S3 path and save it to the input directory
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(airi_path) as resp:
                                if resp.status == 200:
                                    with open(local_image_path, 'wb') as f:
                                        f.write(await resp.read())
                                    logging.info(f"Image successfully downloaded and saved to {local_image_path}")
                                else:
                                    logging.error(f"Failed to download image, status code: {resp.status}")
                                    return web.json_response({"error": "Failed to download image"}, status=500)
                    except Exception as e:
                        logging.error(f"Failed to download image from S3: {str(e)}")
                        return web.json_response({"error": "Failed to download image from S3"}, status=500)

                    # Update the payload with the new image path (GUID)
                    prompt['35']['inputs']['image'] = image_filename

                # Process the prompt
                valid = execution.validate_prompt(prompt)
                if not valid[0]:
                    logging.warning(f"Invalid prompt: {valid[1]}")
                    return web.json_response({"error": valid[1]}, status=400)

                # Generate a unique ID for this prompt
                prompt_id = str(uuid.uuid4())

                number = self.number
                self.number += 1

                # Prepare the extra data if available
                extra_data = json_data.get("extra_data", {})
                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]

                # Add prompt to the queue
                try:
                    outputs_to_execute = valid[2]
                    self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
                except asyncio.QueueFull:
                    logging.error(f"Queue is full, cannot process prompt_id {prompt_id}")
                    return web.json_response({"error": "Server is busy, please try again later"}, status=503)
                logging.info("Prompt successfully submitted to the queue")

                # Establish a WebSocket connection manually
                sid = extra_data.get("client_id", None)
                if not sid:
                    return web.json_response({"error": "client_id is required"}, status=400)

                websocket_url = f"ws://localhost:7860/ws?clientId={sid}"
                logging.info(f"Connecting to WebSocket URL: {websocket_url}")

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(websocket_url) as ws:
                        logging.info(f"Connected to WebSocket with client_id: {sid}")

                        image_filenames = []
                        while True:
                            try:
                                msg = await ws.receive(timeout=600)  # 10 minutes timeout for receiving a message
                                logging.info(f"Received WebSocket message: {msg.data}")

                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    logging.info(f"WebSocket message content: {json.dumps(data, indent=2)}")

                                    if data.get('type') == 'executed' and data['data'].get('prompt_id') == prompt_id:
                                        output = data['data']['output']
                                        if 'images' in output:
                                            if output['images']:  # If images list is not empty
                                                for image_info in output['images']:
                                                    logging.info(f"Image info: {image_info}")
                                                    image_filenames.append(image_info.get('filename'))
                                                break
                                            else:
                                                logging.info(f"Received 'executed' message but images are empty. Waiting for images...")
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    logging.warning(f"WebSocket error: {ws.exception()}")

                            except asyncio.TimeoutError:
                                logging.warning(f"Timeout while waiting for 'executed' message with images for prompt_id {prompt_id}")
                                return web.json_response({"error": "Image generation timed out"}, status=500)

                        if image_filenames:
                            s3_urls = []
                            thumbnail_urls = []
                            image_metadata = []
                            output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output'))
                            for image_filename in image_filenames:
                                path = os.path.join(output_dir, image_filename)

                                # Resize the obtained image to create a thumbnail using Pillow
                                try:
                                    # Open the original image
                                    image = Image.open(path)

                                    # Get image size and dimensions
                                    width, height = image.size
                                    file_size = os.path.getsize(path)

                                    # Store the image metadata
                                    image_metadata.append({
                                        "filename": image_filename,
                                        "width": width,
                                        "height": height,
                                        "size_in_bytes": file_size
                                    })

                                    # Create a thumbnail
                                    image.thumbnail((600, 600))

                                    # Save the thumbnail
                                    thumbnail_filename = "thumbnail_" + image_filename
                                    thumbnail_path = os.path.join(output_dir, thumbnail_filename)
                                    image.save(thumbnail_path)

                                    # Upload both original and thumbnail to S3
                                    s3_client = boto3.client(
                                        's3',
                                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                        region_name=AWS_REGION,
                                    )
                                    new_filename = str(uuid.uuid4()) + '.jpg'
                                    thumbnail_s3_key = f"devEnv/thumbnail_{new_filename}"
                                    original_s3_key = f"devEnv/{new_filename}"

                                    # Upload the original image
                                    s3_client.upload_file(path, AWS_BUCKET_NAME, original_s3_key, ExtraArgs={'ACL': 'public-read'})
                                    s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com.cn/{original_s3_key}"
                                    s3_urls.append(s3_url)

                                    # Upload the thumbnail image
                                    s3_client.upload_file(thumbnail_path, AWS_BUCKET_NAME, thumbnail_s3_key, ExtraArgs={'ACL': 'public-read'})
                                    thumbnail_s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com.cn/{thumbnail_s3_key}"
                                    thumbnail_urls.append(thumbnail_s3_url)

                                    logging.info("Images and thumbnails successfully uploaded to S3")

                                except NoCredentialsError:
                                    logging.error("AWS credentials not available")
                                    return web.json_response({"error": "AWS credentials not available"}, status=500)
                                except ClientError as e:
                                    logging.error(f"Failed to upload image to S3: {str(e)}")
                                    return web.json_response({"error": "Failed to upload image to S3"}, status=500)
                                except Exception as e:
                                    logging.error(f"Failed to process image: {str(e)}")
                                    return web.json_response({"error": "Failed to process the image"}, status=500)

                            return web.json_response({
                                "image_urls": s3_urls, 
                                "image_thumbnail_urls": thumbnail_urls, 
                                "image_metadata": image_metadata
                            })
                        else:
                            logging.error(f"No images found in the output for prompt_id {prompt_id}")
                            return web.json_response({"error": "Failed to generate image"}, status=500)

            except Exception as e:
                logging.error(f"Error in generate_image: {str(e)}")
                logging.error(traceback.format_exc())
                return web.json_response({"error": "Internal server error"}, status=500)


    async def setup(self):
        timeout = aiohttp.ClientTimeout(total=None) # no timeout
        self.client_session = aiohttp.ClientSession(timeout=timeout)

    def add_routes(self):
        self.user_manager.add_routes(self.routes)
        self.app.add_subapp('/internal', self.internal_routes.get_app())

        # Prefix every route with /api for easier matching for delegation.
        # This is very useful for frontend dev server, which need to forward
        # everything except serving of static files.
        # Currently both the old endpoints without prefix and new endpoints with
        # prefix are supported.
        api_routes = web.RouteTableDef()
        for route in self.routes:
            # Custom nodes might add extra static routes. Only process non-static
            # routes to add /api prefix.
            if isinstance(route, web.RouteDef):
                api_routes.route(route.method, "/api" + route.path)(route.handler, **route.kwargs)
        self.app.add_routes(api_routes)
        self.app.add_routes(self.routes)

        for name, dir in nodes.EXTENSION_WEB_DIRS.items():
            self.app.add_routes([
                web.static('/extensions/' + urllib.parse.quote(name), dir),
            ])

        self.app.add_routes([
            web.static('/', self.web_root),
        ])

    def get_queue_info(self):
        prompt_info = {}
        exec_info = {}
        exec_info['queue_remaining'] = self.prompt_queue.get_tasks_remaining()
        prompt_info['exec_info'] = exec_info
        return prompt_info

    async def send(self, event, data, sid=None):
        if event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            await self.send_image(data, sid=sid)
        elif isinstance(data, (bytes, bytearray)):
            await self.send_bytes(event, data, sid)
        else:
            await self.send_json(event, data, sid)

    def encode_bytes(self, event, data):
        if not isinstance(event, int):
            raise RuntimeError(f"Binary event types must be integers, got {event}")

        packed = struct.pack(">I", event)
        message = bytearray(packed)
        message.extend(data)
        return message

    async def send_image(self, image_data, sid=None):
        image_type = image_data[0]
        image = image_data[1]
        max_size = image_data[2]
        if max_size is not None:
            if hasattr(Image, 'Resampling'):
                resampling = Image.Resampling.BILINEAR
            else:
                resampling = Image.ANTIALIAS

            image = ImageOps.contain(image, (max_size, max_size), resampling)
        type_num = 1
        if image_type == "JPEG":
            type_num = 1
        elif image_type == "PNG":
            type_num = 2

        bytesIO = BytesIO()
        header = struct.pack(">I", type_num)
        bytesIO.write(header)
        image.save(bytesIO, format=image_type, quality=95, compress_level=1)
        preview_bytes = bytesIO.getvalue()
        await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, preview_bytes, sid=sid)

    async def send_bytes(self, event, data, sid=None):
        message = self.encode_bytes(event, data)

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_bytes, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_bytes, message)

    async def send_json(self, event, data, sid=None):
        message = {"type": event, "data": data}

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_json, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_json, message)

    def send_sync(self, event, data, sid=None):
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))

    def queue_updated(self):
        self.send_sync("status", { "status": self.get_queue_info() })

    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)

    async def start(self, address, port, verbose=True, call_on_start=None):
        runner = web.AppRunner(self.app, access_log=None)
        await runner.setup()
        ssl_ctx = None
        scheme = "http"
        if args.tls_keyfile and args.tls_certfile:
                ssl_ctx = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS_SERVER, verify_mode=ssl.CERT_NONE)
                ssl_ctx.load_cert_chain(certfile=args.tls_certfile,
                                keyfile=args.tls_keyfile)
                scheme = "https"

        site = web.TCPSite(runner, address, port, ssl_context=ssl_ctx)
        await site.start()

        self.address = address
        self.port = port

        if verbose:
            logging.info("Starting server\n")
            logging.info("To see the GUI go to: {}://{}:{}".format(scheme, address, port))
        if call_on_start is not None:
            call_on_start(scheme, address, port)

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def trigger_on_prompt(self, json_data):
        for handler in self.on_prompt_handlers:
            try:
                json_data = handler(json_data)
            except Exception as e:
                logging.warning(f"[ERROR] An error occurred during the on_prompt_handler processing")
                logging.warning(traceback.format_exc())

        return json_data
