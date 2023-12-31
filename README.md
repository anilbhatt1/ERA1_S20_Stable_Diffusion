## Stable Diffusion

- **Perceptual Loss**
  - Layer-0 of VGG16 model was used as feature extractor
  - It extracts features from both denoised image and the reference image supplied by user
  - Both the features are passed through a MSE loss function so that denoised image will fit towards the refernce image
  - Loss will be applied at fixed intervals like once in 10 iterations on denoised image
  ```
  def perceptual_loss(images, pattern):
    criterion = nn.MSELoss()
    mse_loss = criterion(images, pattern)
    return mse_loss
  ```
  - Loss calculation inside inference is as follows:
  ```
      pattern_loss_scale = 20
      for i, t in tqdm(enumerate(scheduler.timesteps)):
        ...
        ...
        if (i%3 == 0):      
            ...
            ...
              # Calculate loss
            denoised_images_extr = feature_extractor(denoised_images)
            reference_img_extr = feature_extractor(tfm_pattern_image)
            loss = perceptual_loss(denoised_images_extr, reference_img_extr) * pattern_loss_scale
            # Get gradient
            cond_grad = torch.autograd.grad(loss, latents)[0]
            # Modify the latents based on this gradient
            latents = latents.detach() - cond_grad * sigma**2          
  ```

- **ERA1_S20_Stable_Diffusion_with_percept_loss_V1.ipynb**
  - Stable diffusion with manually loading styles & successfully using perception loss
  - Notebook Link : https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/blob/master/ERA1_S20_Stable_Diffusion_with_percept_loss_pipeline_V3.ipynb
- **ERA1_S20_Stable_Diffusion_with_percept_loss_pipeline_V3.ipynb** 
  - Stable diffusion using diffusionpipeline with styles downloaded on-the-fly & successfully using perception loss
  - Notebook Link : https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/blob/master/ERA1_S20_Stable_Diffusion_with_percept_loss_pipeline_V3.ipynb
- **ERA1_S20_Stable_Diffusion_gradio_gpu_inference_V4.ipynb**
  - Gradio inferencing with GPU using diffusionpipeline with styles downloaded on-the-fly & successfully using perception loss
  - Notebook Link : https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/blob/master/ERA1_S20_Stable_Diffusion_gradio_gpu_inference_V4.ipynb
- **app.py** and **requirements.txt**
  - app.py and requirements.txt are hosting the app in Hugging Face CPU space
  - Link for app.py: https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/blob/master/app.py
  - Link for Huggingface App **ImageAlchemy-StableDiffusion**: https://huggingface.co/spaces/anilbhatt1/ImageAlchemy-StableDiffusion
    
### Reference Image supplied for perceptual loss

![wd7](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/ed23b3df-fd29-4d3e-bc98-23eb2387b240)

  
### Results : With style but without perceptual loss
  
  - Prompt Supplied : **A boy running in the style of a tiger**
    
![results_styles_no_loss](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/36b275a8-90a9-420f-8038-469cae1eed7f)

### Results : Without any style but with perceptual loss
  
  - Prompt Supplied : **A boy dreaming gazing at the skies**
    
![results_no_styles_perceptual_loss](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/904241fe-70b2-480e-8d1e-16489ec4b0dc)

### Results : With styles and perceptual loss (manully loading styles)
  
  - Prompt Supplied : **A boy running in the style of a <style>**
    
![results_styles_perceptual_loss](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/4c541536-8578-4d9a-ab4a-de622509acc3)

### Results : With styles and perceptual loss (using diffusion pipeline)
  
  - Prompt Supplied : **A boy running in the style of a <style>**
    
![results_styles_pipeline_perceptual_loss](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/c79ebbac-7bba-499a-afd4-7520c974c3d7)

### Results : Gradio inferencing with GPU T4 (Colab) With styles and perceptual loss (using diffusion pipeline)
  
  - Prompt Supplied : **A toddler gazing at sky in the style of <birb>**

![results_gradio](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/9453bf38-efb6-4f79-81df-300c18981c1b)

### Results for Huggingface App ImageAlchemy-StableDiffusion : https://huggingface.co/spaces/anilbhatt1/ImageAlchemy-StableDiffusion

- **Anime** Results

![results_huggingface_anime](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/4bb3e835-556a-46f3-9f66-0712891df1c7)

- **Depthmap** Results

![results_huggingface_depthmap](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/80b86b18-02da-4bdb-a224-441de0f3eb8a)
  
- **Birb** Results

![results_hf_birb](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/c959246c-b74a-48ff-8258-8d3c69e329be)

- **GTA5** Results

![results_hf_gta5](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/0d5c82c5-a1bc-4e12-be05-9db20648a2da)

- **Midjourney** Results

![results_hf_midjourney](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/7f7a1ec8-dc3f-4ce5-81e1-d36a9602ccad)

- **beetlejuice** Results

![results_hf_beetlejuice](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/be0b8cf1-70a1-4237-be6b-8afc7e149f1e)

- **No style** Results

![results_hf_nostyle](https://github.com/anilbhatt1/ERA1_S20_Stable_Diffusion/assets/43835604/3b393974-0f51-4ba5-8c29-f9f84972ace9)

