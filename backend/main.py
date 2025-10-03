import base64
import uuid
import modal
import os
import requests
import boto3 


from pydantic import BaseModel
from typing import List
from prompts import SONG_DESCRIPTION_PROMPT_GENERATOR, LYRICS_PROMPT_GENERATOR

app = modal.App("synthoria")


# Define a custom image for the Modal environment                                                                 
# This image is based on Debian Slim and includes all necessary dependencies


# Modal employs caching for imagefile if no changes

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/synthesizer-model", "sed -i 's/spacy==3.8.4/spacy/g' /tmp/synthesizer-model/requirements.txt", "cd /tmp/synthesizer-model && pip install ."])

    # Ensures when Modal app downloads a LLM from Hugging Face for the first time, it saves it to a cache
    # On all subsequent runs, the app will find the model in that cache and load it directly 
                                                                 
    .env({"HF_HOME" : "/.cache/huggingface"})
    .add_local_python_source("prompts") 
)

synthoria_secrets = modal.Secret.from_name("synthoria-secret")

# Utilized modal.Volume to ensure qwen-2 and synthesizer model are downloaded & 
# easy to re-distribute w/o re-downloading them again

model_volume = modal.Volume.from_name("synthesizer-model", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-2-hf-cache", create_if_missing=True)



class SynthesizerModelParams(BaseModel):
    audio_duration : float = 180.0
    seed : int = -1
    guidance_scale: float = 15.0
    infer_step: int = 60
    instrumental: bool = False

class SimpleMode(SynthesizerModelParams):
    song_description: str

class CustomModeAutoLyrics(SynthesizerModelParams):
    prompt: str
    lyrics_description: str

class CustomModeManualLyrics(SynthesizerModelParams):
    prompt: str
    lyrics: str


class MusicResponseS3(BaseModel):
    s3_keys: str 
    thumbnail: str
    categories: List[str]


class MusicReponse(BaseModel):
    audio : str


# @app here is a decorator (Functions that take another function as a param)
# @app.cls() for a class & @app.function() for a function
@app.cls(
    image=image, 
    gpu="L40S",
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},  
    secrets= [modal.Secret.from_name("synthoria-secret")],
    scaledown_window= 15,  # Keep container idle for extra 15s after a request is dealt with
                          # If concurrent request, speeds up request response as model already loaded into GPU's memory
    region='us-east-2' 
)
class SynthoriaServer:
    @modal.enter() # Loads model if it was a cold start 
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline            # Importing ACEStepPipeline for loading synthesizer model
        from transformers import AutoModelForCausalLM, AutoTokenizer     # For Qwen-2 Model
        from diffusers import AutoPipelineForText2Image
        import torch

        # synthesizer model - for musics
        self.synthesizer_model = ACEStepPipeline(
            checkpoint_directory = "/models",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )

        # qwen-2 7B model - for lyrics
        qwen2_model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(qwen2_model_id)

        self.qwen2_model = AutoModelForCausalLM.from_pretrained(
            qwen2_model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir = "/.cache/huggingface"
        )

        # sdxl-turbo model - for thumbnails
        self.sdxl_turbo_model = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16", 
            cache_dir = "/.cache/huggingface"
        )

        self.sdxl_turbo_model.to("cuda")     # cuda moves the model into GPU
                                             # normally, sdxl_turbo model is loaded into CPU


    def qwen_2_llm(self, question: str): 

        prompt = question
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.qwen2_model.device)

        generated_ids = self.qwen2_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_audio_tags(self, song_description: str):

        prompt = SONG_DESCRIPTION_PROMPT_GENERATOR.format(user_prompt=song_description) # Insert song description & creates audio tags prompt

        return self.qwen_2_llm(prompt) # Returns audio tags from the prompt
    

    def generate_lyrics(self, lyrics_description: str):

        prompt = LYRICS_PROMPT_GENERATOR.format(descripton=lyrics_description) # Inserts lyric description & creates lyrics generator prompt

        return self.qwen_2_llm(prompt) # Returns lyrics from the prompt
    
    def generate_and_upload_to_s3(
            self,
            prompt: str,
            lyrics: str,
            instrumental: bool,
            audio_duration: float,
            infer_step: int,   # Number of refinement steps (infer_step)
                               # More steps = better quality, slower generation

            guidance_scale: float,  # How much the AI listens to instructions? (guidance_scale)
                                    # Bigger scale = follows input closely, smaller = less tightly bound to prompt, more variations
            seed: int ) -> MusicResponseS3:

            final_lyrics = "[instrumental]" if instrumental else lyrics   

            s3_client = boto3.client("s3")
            bucket_name = os.environ("S3_BUCKET")

            output_dir = "/tmp/outputs"     # Output directory to store music generated temporarily  
            os.makedirs(output_dir, exist_ok=True) 
            output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")       # Using uuid4 for creating random uuid (id) for each song
                                                                                # Avoid uuid1 as it uses computer netwrok address to create uuid, compromising privacy
            self.synthesizer_model(
                prompt= prompt,
                lyrics= final_lyrics,
                audio_duration= audio_duration,
                infer_step= infer_step,
                guidance_scale= guidance_scale,
                save_path= output_path

            )

            audio_s3_key = f"{uuid.uuid4()}.wav"
            s3_client.upload(output_path, bucket_name, audio_s3_key)
            os.remove(output_path)

    

    @modal.fastapi_endpoint(method="POST")  # FastAPI Endpoint
    def synthesize(self) -> MusicReponse:
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

        self.synthesizer_model(
            prompt="",
            lyrics= "",
            audio_duration=180,
            infer_step=60,
            guidance_scale=50,
            save_path=output_path

        )

        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8") # Turns audio to string to make JSON Compatible   
        os.remove(output_path)

        return MusicReponse(audio=audio_base64)
    

    @modal.fastapi_endpoint(method="POST")
    def simple_mode(self, request: SimpleMode) -> MusicResponseS3:
        # Generate audio tags
        audio_tags = self.generate_audio_tags(request.song_description)

        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_lyrics(request.song_description)  # Generate lyrics 
        

    @modal.fastapi_endpoint(method="POST")
    def custom_mode_auto_lyric(self, request : CustomModeAutoLyrics) -> MusicResponseS3:
        pass

    @modal.fastapi_endpoint(method="POST")
    def custom_mode_manual_lyric(self, request : CustomModeManualLyrics) -> MusicResponseS3:
        pass


@app.local_entrypoint()
def main():
    server = SynthoriaServer()
    endpoint_url = server.synthesize.get_web_url()
    response = requests.post(endpoint_url)

    response.raise_for_status()
    result = MusicReponse(**response.json())

    audio_bytes = base64.b64decode(result.audio)
    output_filename = "generated.wav"

    with open(output_filename, "wb") as f:
        f.write(audio_bytes)

    

    

        















