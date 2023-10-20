import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as tfms
import torchvision.models as models

from PIL import Image
import numpy as np
from diffusers import LMSDiscreteScheduler, DiffusionPipeline

import random
import os
import subprocess

from matplotlib import pyplot as plt
from pathlib import Path
from torch import autocast
from tqdm.auto import tqdm

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a pre-trained VGG model (you can use other models as well)
vgg_model = models.vgg16(pretrained=True).features
vgg_model = vgg_model.to(torch_device)

# Create a new model that extracts features from the chosen layers
feature_extractor = nn.Sequential()
for name, layer in vgg_model._modules.items():
    if name == '0':  # Stop at the 0th layer
        break
    feature_extractor.add_module(name, layer)
feature_extractor = feature_extractor.to(torch_device)

pretrained_model_name_or_path = "segmind/tiny-sd"
pipe = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.float32
).to(torch_device)

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

concept_dict={'anime_bg_v2':('sd-concepts-library/anime-background-style-v2','<anime-background-style-v2>',31),
              'birb':('sd-concepts-library/birb-style','<birb-style>',32),
              'depthmap':('sd-concepts-library/depthmap','<depthmap>',33),
              'gta5_artwork':('sd-concepts-library/gta5-artwork','<gta5_artwork>',34),
              'midjourney':('sd-concepts-library/midjourney-style','<midjourney-style>',35),
              'beetlejuice':('sd-concepts-library/beetlejuice-cartoon-style','<beetlejuice-cartoon>',36)}

cache_style_list = []

def transform_pattern_image(pattern_image):
    preprocess = tfms.Compose([
        tfms.Resize((320, 320)),
        tfms.ToTensor(),
    ])
    tfms_pattern_image = preprocess(pattern_image).unsqueeze(0)
    return tfms_pattern_image

def load_required_style(style):
    for concept, value in concept_dict.items():
        if style in concept:
            concept_key = value[1]
            concept_seed = value[2]
            if style not in cache_style_list:
                pipe.load_textual_inversion(value[0])
                cache_style_list.append(style)
            break
    return concept_key, concept_seed

def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = pipe.vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()  # [1, 4, 64, 64]

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def perceptual_loss(images, pattern):
    """
    This function calculates the perceptual loss between the output image and the target image.

    Parameters:
    """
    criterion = nn.MSELoss()
    mse_loss = criterion(images, pattern)
    return mse_loss

#Generating image with the modified embeddings with pattern loss guidance and saving the images to steps/{concept} folder
def generate_with_embs_pattern_loss(prompt, concept_seed, tfm_pattern_image, num_inf_steps):
    height = 320                        # default height of Stable Diffusion
    width = 320                         # default width of Stable Diffusion
    num_inference_steps = num_inf_steps # Number of denoising steps
    guidance_scale = 8                  # Scale for classifier-free guidance
    generator = torch.manual_seed(concept_seed) # Seed generator to create the inital latent noise
    batch_size = 1
    pattern_loss_scale = 20

    text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    input_ids = text_input.input_ids.to(torch_device)
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = pipe.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    latents = torch.randn((batch_size, pipe.unet.in_channels, height // 8, width // 8),
                           generator=generator,)
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # Loop
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform CFG (Classifier Free Guidance)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        #### ADDITIONAL GUIDANCE ###
        if (i%3 == 0):
            # Requires grad on the latents
            latents = latents.detach().requires_grad_()

            # Get the predicted x0:
            latents_x0 = latents - sigma * noise_pred
            # latents_x0 = scheduler.step(noise_pred, t, latents).pred_original_sample

            # Decode to image space
            denoised_images = pipe.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 # range (0, 1)
            # Calculate loss
            denoised_images_extr = feature_extractor(denoised_images)
            reference_img_extr = feature_extractor(tfm_pattern_image)
            loss = perceptual_loss(denoised_images_extr, reference_img_extr) * pattern_loss_scale
            # Get gradient
            cond_grad = torch.autograd.grad(loss, latents)[0]

            # Modify the latents based on this gradient
            latents = latents.detach() - cond_grad * sigma**2

        # Now step with scheduler. compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents

def generate_image(prompt, pattern_image, style, num_inf_steps):
    tfm_pattern_image = transform_pattern_image(pattern_image)  # Transform the pattern image to be fed to feature extractor
    tfm_pattern_image = tfm_pattern_image.to(torch_device)
    if style == "no-style":
        concept_seed = 40
        main_prompt = str(prompt)
    else:
        concept_key, concept_seed = load_required_style(style)
        main_prompt = f"{str(prompt)} in the style of {concept_key}"
    latents = generate_with_embs_pattern_loss(main_prompt, concept_seed, tfm_pattern_image, num_inf_steps)
    generated_image = latents_to_pil(latents)[0]
    return generated_image

def gradio_fn(prompt, pattern_image, style, num_inf_steps):
    output_pil_image = generate_image(prompt, pattern_image, style, num_inf_steps)
    return output_pil_image

demo = gr.Interface(fn=gradio_fn,
                    inputs=[gr.Textbox(info="Example prompt: 'A toddler gazing at sky'"),
                            gr.Image(type="pil", height=224, width=224, info='Sample image to emulate the pattern'),
                            gr.Radio(["anime","birb","depthmap","gta5","midjourney","beetlejuice","no-style"], label="Style",
                                     info="Choose the style in which image to be made"),
                            gr.Slider(50, 200, value=50, label="Num_inference_steps", info="Choose between 50, 10, 150 & 200")],
                    outputs=gr.Image(height=320, width=320),
                    title="ImageAlchemy using Stable Diffusion",
                    description="- Stable Diffusion model that generates single image to fit \
                                  (a) given text prompt (b) given reference image and (c) selected style.")

demo.launch(share=True)