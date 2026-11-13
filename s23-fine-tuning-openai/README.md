# Section 23: Fine-tuning LLMs

Fine-tune a small open-weights chat model (Qwen3-1.7B, Apache-2.0) on TeaCrafters support conversations,
then run it on your own machine with Ollama. No API, no cutoff dates.

| File | What it does | Where it runs |
|---|---|---|
| `teacrafters_policies.md` | The only facts the support bot may state | — |
| `01_prepare_data.py` | Drafts ~100 example conversations with Claude Haiku, validates them, writes train/test JSONL | Your machine, **Setup B (uv)** |
| `teacrafters_train.jsonl` / `teacrafters_test.jsonl` | The dataset (OpenAI-style `messages` JSONL), ready to use | — |
| `02_finetune_colab.ipynb` | LoRA fine-tune with Unsloth on a free T4, export GGUF + Ollama `Modelfile` | [Google Colab](https://colab.research.google.com/github/pdichone/ai-llm-engineer-course/blob/main/s23-fine-tuning-openai/02_finetune_colab.ipynb) |
| `03_chat_ollama.py` | Base vs fine-tuned on unseen questions, then `--chat` | Your machine, **Setup B (uv)** + Ollama |

## Run it

```bash
# from the repo root
uv run s23-fine-tuning-openai/01_prepare_data.py      # optional: the JSONL files are already here
# open 02_finetune_colab.ipynb in Colab, run all cells, download teacrafters_ollama.zip, unzip it, then:
ollama create teacrafters -f Modelfile
ollama pull qwen3:1.7b
uv run s23-fine-tuning-openai/03_chat_ollama.py
```

**Notes**
- Current Ollama no longer loads LoRA adapters on their own: merge first (the notebook does it for you).
- Qwen3 "thinks" by default. The notebook trains the non-thinking format, and the script calls Ollama with `think=False`.
- On an Apple Silicon Mac you can train natively instead of Colab:

  ```bash
  mlx_lm.lora --model Qwen/Qwen3-1.7B --train --data <folder with train.jsonl> --iters 36 --mask-prompt
  mlx_lm.fuse --model Qwen/Qwen3-1.7B --save-path fused_model
  ```

  Then use a Modelfile containing just `FROM ./fused_model`.

## Legacy: OpenAI fine-tuning (watch only)
`ft_openai.py`, `teacrafter.json` and `output.jsonl` go with the original OpenAI walkthrough lectures.
OpenAI has closed fine-tuning to new organizations. No new jobs can be created after January 6, 2027, and
OpenAI fine-tuned models stop working on January 14, 2027. Run it with **Setup A (classic)**:
`cd s23-fine-tuning-openai && python ft_openai.py`.

Back to the [course index](../README.md).
