There are three notebooks in this repository:
1) **Pretraining and Fine tuning Notebook** Used PyTorch and Open AI's early Open Source Model GPT2 to develop a deeper understanding and intuition of how language models are trained. We will look at a specific simple task, Sentiment Classification, and see in two ways how we can use language models for this problem.
a. **Continued GPT-2 Pretraining of GPT with a Movie Plots dataset**  
Explore how continued pretraining affects a Language Model. This gives us a good idea how pretraining morks, and more specifically, how additional
domain-specific pretraining affects the perplexity for text within this domain vs outside of the domain. 
b. **Fine-tuning of GPT-2 for Sentiment Analysis of the IMDB dataset (using Pre/Post-Modifiers)**
Used both the original GPT-2 model and the model that was further pretrained on the CMU Movie Summary dataset for a Sentiment Analysis of the IMDB movie review dataset,
which is part of Hugging Face datasets. We will (hopefully(!)... there are statistical fluctuations) see that fine-tuning the model that was further pre-trained on the movie
plot dataset behaves somewhat better than the original gpt-2 model fine-tuned.
c.  **IMDB Sentiment Classification with a Masked Language Model (BERT)**
   We will also look at a Masked Language Model, specifically BERT, as a tool for Sentiment Classification of the same dataset.


2) **Fine Tuning Notebook** This notebook uses PyTorch and Open AI's early Open Source Model GPT2 to develop a deeper understanding and intuition of how language models are fine-tuned with parameter efficient fine tuning (PEFT) techniques. We will continue to look at a specific simple task, Sentiment Classification, and see how we can fine tune two different models to improve performance.
a. **Fine-tuning of [GPT2 large](https://huggingface.co/openai-community/gpt2-large) with a sentiment classification dataset**  
Explored how we can combine Low Rank Adaptation (LoRA) with Quantization to fine tune a larger model.
b. **Fine-tuning of Gemma model for Sentiment Analysis**  
 Used a larger more recent model -- [Gemma 2](https://huggingface.co/google/gemma-2b) from Google
 -- to illustrate the benefits of an increase in the number of parameters and how it affects the performance of the model.  
 This model doubles the number or parameters but has also undergone a better pre-training regime and we would expect that to be reflected in the performance of the model.


3) **Prompt Engineering Notebook** # Used PyTorch and Hugging Face to:
   a. **Generating images with [Stable Diffusion 1](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) using text prompts**  
   Explore how we can use text to instruct a model to generate an image. We'll see how well the model can follow our instructions by asking it to produce specific numbers of
   objects.  We'll then use a different model to see if the numbers of objects produced is correct.
   b. **Examining the capabilities of Mistral**  
   Use a recent large language model -- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) from Mistral -- to illustrate the benefits of an increase
   in the number of parameters and how it affects the performance of the model.  This model doubles the number or parameters but has also undergone a better pre-training and
   post-training regime and we would expect that to be reflected in the performance of the model and its ability to follow instructions.

