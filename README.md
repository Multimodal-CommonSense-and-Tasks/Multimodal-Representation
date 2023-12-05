# Multimodal-Representation

#### This repository organizes researches related to AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data, especially Multimodal-Representation task. 
#### This repository summarizes following researches.

## Research list
* Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models (ICML 2023 Oral) - Dongjun Kim, Yeongmin Kim, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon.

  * The proposed Discriminator Guidance (DG), aims to improve sample generation of pre-trained diffusion models. The approach introduces a discriminator that gives explicit supervision to a denoising sample path whether it is realistic or not.

* Maximum Likelihood Training of Implicit Nonlinear Diffusion Model (NeurIPS 2022 Poster) - Dongjun Kim, Byeonghu Na, Se Jung Kwon, Dongsoo Lee, Wanmo Kang, and Il-Chul Moon.

  * The proposed Implicit Nonlinear Diffusion Model (INDM) learns by combining a normalizing flow and a diffusion process. INDM implicitly constructs a nonlinear diffusion on the data space by leveraging a linear diffusion on the latent space through a flow network with connecting by invertible function.

* Loss Curvature Matching for Dataset Selection and Condensation (AISTATS 2023 Poster) - Seungjae Shin, HeeSun Bae, Donghyeok Shin, Weonyoung Joo, and Il-Chul Moon.

  * The proposed Loss Curvature Matching (LCMat), is a new reduction objective matching the loss curvatures of the original dataset and reduced dataset over the model parameter space, more than the parameter point. This new objective induces a better adaptation of the reduced dataset on the perturbed parameter region than the exact point matching.

* From Noisy Prediction to True Label: Noisy Prediction Calibration via Generative Model (ICML 2022) - HeeSun Bae, Seungjae Shin, Byeonghu Na, JoonHo Jang, Kyungwoo Song, and Il-Chul Moon.

  * The proposed Noise Prediction Calibration (NPC), calibrates the prediction from a imperfectly pre-trained classifier to a true label via utilizing a deep generative model. NPC operates as a post-processing module to a black-box classifier, without further access into classifier parameters.

* SAAL: Sharpness-Aware Active Learning (ICML 2023) - Yoon-Yeong Kim, Youngjae Cho, Joonho Jang, Byeonghu Na, Yeongmin Kim, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon.

  * The proposed Sharpness-Aware Active Learning (SAAL), is the first active learning method to incorporate the sharpness of loss space into the acquisition function. SAAL constructs its acquisition function by selecting unlabeled instances whose perturbed loss becomes maximum.

* Soft Truncation: A Universal Training Technique of Score-based Diffusion Model for High Precision Score Estimation (ICML 2022 Poster) - Dongjun Kim,Seungjae Shin, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon.

  * The proposed Soft Truncation (ST), is a universally applicable training technique for diffusion models, softens the fixed and static truncation hyperparameter into a random variable. ST softens the truncation level at each mini-batch update, and this simple modification is connected to the general weighted diffusion loss and the concept of Maximum Perturbed Likelihood Estimation.

## Acknowledgements
These works were supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT). Also, these works were supported by supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT).
