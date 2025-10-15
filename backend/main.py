import base64
import uuid
import modal
import os
import requests
import boto3 


from pydantic import BaseModel
from typing import List
from prompts import CATEGORY_PROMPT, SONG_DESCRIPTION_PROMPT_GENERATOR, LYRICS_PROMPT_GENERATOR, THUMBNAIL_PROMPT_TEMPLATE

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
    audio_s3_key: str 
    thumbnail_s3_key: str
    categories: List[str]


class MusicReponse(BaseModel):
    audio : str


# @app here is a decorator (Functions that take another function as a param)
# @app.cls() for a class & @app.function() for a function
@app.cls(
    image=image, 
    gpu="L40S",     # Outperforms A100 on AI inference
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},        # Volumes store the music-generation and hugging faces' model (qwen2b & sdxl_turbo)
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

    def generate_audio_prompt(self, song_description: str):

        prompt = SONG_DESCRIPTION_PROMPT_GENERATOR.format(user_prompt=song_description) # Insert song description & creates prompt for the audio

        return self.qwen_2_llm(prompt) 
    

    def generate_lyrics(self, lyrics_description: str):

        prompt = LYRICS_PROMPT_GENERATOR.format(description=lyrics_description) # Inserts lyric description & creates lyrics generator prompt

        return self.qwen_2_llm(prompt) # Returns lyrics from the prompt
    
    def generate_category_tags(self, description: str) -> List[str]:

        prompt = CATEGORY_PROMPT.format(description=description)        # Passing description of songs to generate tags to categorize audio
        category_response =  self.qwen_2_llm(prompt)
        categories_list = [category.strip() for category in category_response.split(",") if category.strip()]
        return categories_list


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
            seed: int ,
            description_for_categorization: str

            ) -> MusicResponseS3:

            final_lyrics = "[instrumental]" if instrumental else lyrics   

            s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
                aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
                region_name=os.environ["AWS_REGION"]
            )

            bucket_name = os.environ["S3_BUCKET"]

            output_dir = "/tmp/outputs"     # Output directory to store music generated temporarily  
            os.makedirs(output_dir, exist_ok=True) 
            output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")       # Using uuid4 for creating random uuid (filename) for each song
                                                                                # Avoid uuid1 as it uses computer netwrok address to create uuid, compromising privacy
            self.synthesizer_model(
                prompt= prompt,
                lyrics= final_lyrics,
                audio_duration= audio_duration,
                infer_step= infer_step,
                guidance_scale= guidance_scale,
                save_path= output_path,
                manual_seeds = str(seed)

            )

            """Upload music to S3"""

            audio_s3_key = f"{uuid.uuid4()}.wav"
            s3_client.upload_file(output_path, bucket_name, audio_s3_key)
            os.remove(output_path)


            """Generate and upload thumbnail to S3"""

            thumbnail_prompt  = THUMBNAIL_PROMPT_TEMPLATE.format(prompt=prompt)
            thumbnail = self.sdxl_turbo_model(prompt= thumbnail_prompt, num_inference_steps=2, guidance_scale=0.0).images[0] 
            thumbnail_output_path = os.path.join(output_dir, f"{uuid.uuid4()}.png")
            thumbnail.save(thumbnail_output_path)
            
            thumbnail_s3_key = f"{uuid.uuid4()}.png"
            s3_client.upload_file(thumbnail_output_path, bucket_name, thumbnail_s3_key)
            os.remove(thumbnail_output_path)

            """Audio Categories"""

            categories = self.generate_category_tags(description_for_categorization)

            return MusicResponseS3(                 # The generated keys and categories are returned to store them into the database
                audio_s3_key = audio_s3_key,
                thumbnail_s3_key = thumbnail_s3_key,
                categories = categories
            )

    @modal.fastapi_endpoint(method="POST")
    def simple_mode(self, request: SimpleMode) -> MusicResponseS3:

        audio_prompt = self.generate_audio_prompt(request.song_description)     # The prompt is just for generating music

        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_lyrics(request.song_description) 
        
        return self.generate_and_upload_to_s3(
            prompt=audio_prompt, 
            lyrics=lyrics, 
            description_for_categorization= request.song_description,
            **request.model_dump(exclude={"song_description"})    # model_dump jsonify the 
                                    
        )


    @modal.fastapi_endpoint(method="POST")
    def custom_mode_auto_lyric(self, request : CustomModeAutoLyrics) -> MusicResponseS3:
    
        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_lyrics(request.lyrics_description) 
        
        return self.generate_and_upload_to_s3(
            prompt=request.prompt, 
            lyrics=lyrics, 
            description_for_categorization= request.prompt,
            **request.model_dump(exclude={"prompt", "lyrics_description"})    # model_dump() for jsonifying the requests' attributes
                                    
        )

    @modal.fastapi_endpoint(method="POST")
    def custom_mode_manual_lyric(self, request : CustomModeManualLyrics) -> MusicResponseS3:

        return self.generate_and_upload_to_s3(
            prompt=request.prompt, 
            lyrics=request.lyrics, 
            description_for_categorization= request.prompt,
            **request.model_dump(exclude={'prompt', 'lyrics'})                               
        ) 


# Endpoint tests (Not to be included in the final code !!!)

@app.local_entrypoint()
def main():
    server = SynthoriaServer()
    endpoint_url = server.custom_mode_manual_lyric.get_web_url()
    
    request_data = CustomModeManualLyrics(
        prompt="Rap, 90s, Old School, Westside",
        lyrics = """[Intro: Roger Troutman]
                    California love, we-ooh

                    [Chorus: Roger Troutman]
                    California knows how to party
                    California knows how to party
                    In the city of L.A. 
                    In the city of good ol' Watts
                    In the city, city of Compton
                    We keep it rockin', we keep it rockin' (Ooh)

                    [Verse 1: Dr. Dre]
                    Now, let me welcome everybody to the Wild, Wild West
                    A state that's untouchable like Eliot Ness
                    The track hits your eardrum like a slug to your chest
                    Pack a vest for your Jimmy in the city of sex
                    We in that sunshine state where the bomb-ass hemp be
                    The state where you never find a dance floor empty
                    And pimps be on a mission for them greens
                    Lean mean money-makin'-machines servin' fiends
                    I've been in the game for ten years makin' rap tunes
                    Ever since honeys was wearin' Sassoon
                    Now it's '95 and they clock me and watch me
                    Diamonds shinin', lookin' like I robbed Liberace
                    It's all good, from Diego to the Bay
                    Your city is the bomb if your city makin' pay (Uh)
                    Throw up a finger if you feel the same way
                    Dre puttin' it down for Californ-i-a
                  
                    [Chorus: Roger Troutman & Dr. Dre]
                    California (California) knows how to party (Knows how to party)
                    California (West Coast) knows how to party (Yes, they do, that's right)
                    In the city of L.A. (City of L.A.; Los Angeles)
                    (Yeah) In the city of good ol' Watts (Good ol' Watts)
                    In the city, city of Compton (City of Compton)
                    Keep it rockin' (Keep it rockin'), keep it rockin' (Come on, come on, come on)
                    Yeah, now make it shake, c'mon

                    [Post-Chorus: Roger Troutman & Dr. Dre]
                    Shake it, shake it, baby (Come on, come on, come on)
                    Shake, shake it (Shake it, baby)
                    Shake, shake it, mama (Come on, come on, come on)
                    Shake it, Cali (Come on, come on, shake it, Cali)
                    Shake it (We don't care), shake it, baby (Baby, right; that's right, uh)
                    Shake it, shake it (Shake, shake, shake, shake)
                    Shake it, shake it, mama (Shake, shake it, shake, shake)
                    Shake it, Cali (Shake it now)

                    [Verse 2: 2Pac & Dr. Dre]
                    Out on bail, fresh out of jail, California dreamin'
                    Soon as I step on the scene, I'm hearin' hoochies screamin'
                    Fiendin' for money and alcohol, the life of a Westside player
                    Where cowards die and the strong ball
                    Only in Cali, where we riot, not rally, to live and die
                    In L.A., we wearin' Chucks, not Ballys (Yeah, that's right, uh)
                    Dressed in Locs and Khaki suits and ride is what we do
                    Flossin' but have caution, we collide with other crews
                    Famous because we throw grams
                    Worldwide, let 'em recognize from Long Beach to Rosecrans
                    Bumpin' and grindin' like a slow jam
                    It's Westside, so you know the Row won't bow down to no man
                    Say what you say, but give me that bomb beat from Dre
                    Let me serenade the streets of L.A. 
                    From Oakland to Sac-town, the Bay Area and back down
                    Cali is where they put they mack down, give me love
                    [Chorus: Roger Troutman & Dr. Dre]
                    California (California) knows how to party (Ain't no stoppin')
                    California (Do-do-do-do-do) knows how to party (Come on, baby)
                    In the city (South Central) of L.A. (L.A.)
                    In the city of good ol' Watts (That's right)
                    In the city, city of Compton (Yup, yup)
                    We keep it rockin' (Keep it rockin'), we keep it rockin' (Yeah, yeah)
                    Now make it shake, uh

                    [Post-Chorus: Roger Troutman & Dr. Dre]
                    Shake, shake it, baby (Uh, shake)
                    Shake, shake it (Uh, yeah; shake it, Cali)
                    Shake it, shake it, mama (Yeah)
                    Shake it, Cali (Shake it, Cali)
                    Shake it, shake it, baby (Shake it, Cali)
                    Shake it, shake it (Uh, uh)
                    Shake it, shake it, mama (West Coast)
                    Shake it, Cali (Uh)

                    [Outro: 2Pac, Dr. Dre & Roger Troutman]
                    Yeah, uh
                    Uh, Long Beach in the house, uh, yeah
                    Oaktown
                    Oakland definitely in the house (Ha, ha-ha-ha-ha)
                    Frisco, Frisco (Yeah)
                    Hey, you know L.A. up in this
                    Pasadena, where you at?
                    Yeah, Inglewood
                    Inglewood always up to no good
                    Even Hollywood are tryna get a piece, baby
                    Sacramento, Sacramento, where you at? Uh, yeah
                    Throw it up, y'all, throw it up, throw it up
                    I can't see ya
                    California love
                    Let's show these fools how we do it on this Westside
                    'Cause you and I know it's the best side"""
    )

    payload = request_data.model_dump()             

    response = requests.post(endpoint_url, json=payload)
    response.raise_for_status()
    result = MusicResponseS3(**response.json())

    print(f"Success.\n Audio S3 Key: {result.audio_s3_key}, Thubmnail S3 Key: {result.thumbnail_s3_key}, and Category: {result.categories}")


    

    

        















