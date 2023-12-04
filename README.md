# Multimodal-Representation

### This repository organizes researches related to AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data, especially Multimodal-Representation task. This repository summarizes following researches.

## Research list
* Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models (DG) (ICML 2023 Oral) - Dongjun Kim, Yeongmin Kim, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon.
  * The proposed method, Discriminator Guidance, aims to improve sample generation of pre-trained diffusion models. The approach introduces a discriminator that gives explicit supervision to a denoising sample path whether it is realistic or not. Unlike GANs, our approach does not require joint training of score and discriminator networks. Instead, we train the discriminator after score training, making discriminator training stable and fast to converge. In sample generation, we add an auxiliary term to the pre-trained score to deceive the discriminator. This term corrects the model score to the data score at the optimal discriminator, which implies that the discriminator helps better score estimation in a complementary way.
 
* Maximum Likelihood Training of Implicit Nonlinear Diffusion Model (INDM) (NeurIPS 2022 Poster) - Dongjun Kim, Byeonghu Na, Se Jung Kwon, Dongsoo Lee, Wanmo Kang, and Il-Chul Moon.
  * Explanation
* Loss Curvature Matching for Dataset Selection and Condensation (LCMat) (AISTATS 2023 Poster) - Seungjae Shin, HeeSun Bae, Donghyeok Shin, Weonyoung Joo, and Il-Chul Moon.
  * Explanation 
* From Noisy Prediction to True Label: Noisy Prediction Calibration via Generative Model (NPC) (ICML 2022) - HeeSun Bae, Seungjae Shin, Byeonghu Na, JoonHo Jang, Kyungwoo Song, and Il-Chul Moon.
  * Explanation
* SAAL: Sharpness-Aware Active Learning (ICML 2023) - Yoon-Yeong Kim, Youngjae Cho, Joonho Jang, Byeonghu Na, Yeongmin Kim, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon.
  * Explanation
* Soft Truncation: A Universal Training Technique of Score-based Diffusion Model for High Precision Score Estimation (ICML 2022 Poster) - Dongjun Kim,Seungjae Shin, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon
  * Explanation

## Acknowledgements
These works were supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT). Also, these works were supported by supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government(MSIT).
