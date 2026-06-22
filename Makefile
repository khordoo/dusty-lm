EPOCHS ?= 23
CHECKPOINT_EVERY_STEPS ?= 100
CHECKPOINT_STEP ?=
PROMPT ?= i wake up.
PROFILE ?= dusty8m
DUSTY_MODEL ?= qwen/qwen3-235b-a22b-2507:floor
DUSTY_FALLBACK_MODEL ?= openai/gpt-oss-120b:floor
DUSTY_SFT_PER_CATEGORY ?= 500
DUSTY_SFT_BATCH_SIZE ?= 20
DUSTY_PRETRAIN_WORKERS ?= 5
DUSTY_PRETRAIN_OUT ?= artifacts/datasets/dusty_pretrain.txt
DUSTY_PRETRAIN_PROGRESS ?= artifacts/datasets/dusty_pretrain_progress.txt
DUSTY_SFT_OUT ?= artifacts/datasets/dusty_sft.jsonl
DUSTY_SFT_REJECTED ?= artifacts/datasets/dusty_sft_rejected.jsonl
DUSTY_SFT_FILTERED_OUT ?= artifacts/datasets/dusty_sft_2000.jsonl
DUSTY_SFT_FILTER_TARGET ?= 2000
DUSTY_SFT_MAX_ANSWER_TOKENS ?= 256
DUSTY_SFT_SAMPLING_MODE ?= balanced
DUSTY_SFT_CHATML_PRETRAIN_OUT ?= artifacts/datasets/dusty_sft_chatml_pretrain.txt

.PHONY: help dusty-generate-pretrain dusty-generate-sft dusty-filter-sft dusty-flatten-sft-pretrain dusty-tokenizer dusty-pretrain-data dusty-pretrain dusty-generate dusty-sft-data dusty-sft-train tensorboard

help:
	@echo "TinyGPT commands"
	@echo ""
	@echo "Dusty 8M workflow:"
	@echo "  make dusty-generate-pretrain   Generate artifacts/datasets/dusty_pretrain.txt"
	@echo "  make dusty-generate-sft        Generate artifacts/datasets/dusty_sft.jsonl"
	@echo "  make dusty-filter-sft          Filter/sample SFT JSONL to artifacts/datasets/dusty_sft_2000.jsonl"
	@echo "  make dusty-flatten-sft-pretrain Flatten SFT JSONL to ChatML pretrain text"
	@echo "  make dusty-tokenizer            Train artifacts/tokenizers/dusty_tokenizer.json"
	@echo "  make dusty-pretrain-data        Tokenize artifacts/datasets/dusty_pretrain.txt"
	@echo "  make dusty-pretrain EPOCHS=1    Train the dusty8m profile"
	@echo "  make dusty-sft-data             Tokenize artifacts/datasets/dusty_sft.jsonl"
	@echo "  make dusty-sft EPOCHS=1         Train the sft_dusty8m profile"
	@echo "  make dusty-generate             Generate with PROFILE=dusty8m PROMPT='i wake up.'"
	@echo "  make tensorboard                Plot training logs from runs/"

dusty-generate-pretrain:
	uv run python dataset_generation/generate_pretrain_dataset_gen_async.py \
		--model $(DUSTY_MODEL) \
		--workers $(DUSTY_PRETRAIN_WORKERS) \
		--out $(DUSTY_PRETRAIN_OUT) \
		--progress $(DUSTY_PRETRAIN_PROGRESS)

dusty-generate-sft:
	uv run python dataset_generation/generate_sft_dataset_with_fallback.py \
		--mode generate \
		--model $(DUSTY_MODEL) \
		--fallback-model $(DUSTY_FALLBACK_MODEL) \
		--per-category $(DUSTY_SFT_PER_CATEGORY) \
		--batch-size $(DUSTY_SFT_BATCH_SIZE) \
		--temperature 0.8 \
		--max-empty-batches 3 \
		--max-user-occurrences 5 \
		--acceptance-window 5 \
		--sleep 0.2 \
		--out $(DUSTY_SFT_OUT) \
		--rejected $(DUSTY_SFT_REJECTED)

dusty-filter-sft:
	uv run python dataset_generation/filter_sft_dataset.py \
		--input $(DUSTY_SFT_OUT) \
		--output $(DUSTY_SFT_FILTERED_OUT) \
		--target-total $(DUSTY_SFT_FILTER_TARGET) \
		--max-answer-tokens $(DUSTY_SFT_MAX_ANSWER_TOKENS) \
		--sampling-mode $(DUSTY_SFT_SAMPLING_MODE)

dusty-flatten-sft-pretrain:
	uv run python dataset_generation/flatten_sft_to_pretrain.py \
		--input $(DUSTY_SFT_OUT) \
		--output $(DUSTY_SFT_CHATML_PRETRAIN_OUT)

dusty-tokenizer:
	uv run python -m tiny_gpt.tokenizer

dusty-pretrain-data:
	uv run python -m tiny_gpt.data_prep --profile dusty8m

dusty-pretrain:
	uv run python -m tiny_gpt.train --profile dusty8m --epochs $(EPOCHS) --checkpoint-every-steps $(CHECKPOINT_EVERY_STEPS)

dusty-generate:
	uv run python tiny_gpt/generate.py --profile $(PROFILE) --prompt "$(PROMPT)" $(if $(CHECKPOINT_STEP),--checkpoint-step $(CHECKPOINT_STEP),)

dusty-sft-data:
	uv run python -m tiny_gpt.data_prep --profile sft_dusty8m

dusty-sft-train:
	uv run python -m tiny_gpt.train --profile sft_dusty8m --epochs $(EPOCHS) --checkpoint-every-steps $(CHECKPOINT_EVERY_STEPS)

tensorboard:
	uv run tensorboard --logdir runs
