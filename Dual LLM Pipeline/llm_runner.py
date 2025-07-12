# from llama_cpp import Llama
# import gc 

# def run_llm_on_all_models(prompt, context, available_models):
#     results = {}
#     input_text = f"{prompt}\n\n{context}"

#     for model_name, model_path in available_models.items():
#         print(f"🧠 Running model: {model_name}")

#         try:
#             llm = Llama(

#                 model_path=model_path,
#                 n_gpu_layers=-1,
#                 n_threads=16,
#                 n_ctx=12000,
#                 temp=0.6,
#                 seed=3407,
#                 tensor_split=[0.5, 0.5],
#             )

#             response = llm.create_chat_completion(
#                 messages=[
#                     {"role": "system", "content": "You extract Bowtie diagram JSON from technical context."},
#                     {"role": "user", "content": input_text}
#                 ],
#                 temperature=0.6,
#                 max_tokens=10000
#             )

#             results[model_name] = response["choices"][0]["message"]["content"]

#         except Exception as e:
#             results[model_name] = f"❌ Model Error: {str(e)}"

#         finally:
#             if 'llm' in locals():
#                 try:
#                     llm.__del__()  # Force llama.cpp cleanup
#                 except:
#                     pass
#                 del llm
#                 gc.collect()

#     return results

from llama_cpp import Llama
import gc


def run_llm_on_all_models(prompt, context, available_models, seeds):
    """
    Run LLM inference for each model using a shared list of random seeds.
    
    Args:
        prompt (str): The prompt template.
        context (str): The FMEA or input context.
        available_models (dict): {model_name: model_path}
        seeds (List[int]): List of seeds to use per run. Length = n_runs.

    Returns:
        Dict[str, List[Dict]]: {model_name: [{"seed": ..., "output": ...}, ...]}
    """
    results = {}
    input_text = f"{prompt}\n\n{context}"

    for model_name, model_path in available_models.items():
        print(f"🧠 Running model: {model_name}")

        model_results = []

        for i, seed in enumerate(seeds):
            print(f"  🔁 Run {i+1} | Seed: {seed}")
            try:
                llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,
                    n_threads=16,
                    n_ctx=12000,
                    temp=0.7,
                    seed=seed,
                )

                response = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You extract Bowtie diagram JSON from technical context."},
                        {"role": "user", "content": input_text}
                    ],
                    temperature=0.7,
                    max_tokens=10000
                )

                model_results.append({
                    "seed": seed,
                    "output": response["choices"][0]["message"]["content"]
                })

            except Exception as e:
                model_results.append({
                    "seed": seed,
                    "output": f"❌ Model Error: {str(e)}"
                })

            finally:
                if 'llm' in locals():
                    try:
                        llm.__del__()
                    except:
                        pass
                    del llm
                    gc.collect()

        results[model_name] = model_results

    return results
