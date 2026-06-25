EPOCHS ?= 23
CHECKPOINT_EVERY_STEPS ?= 100
CHECKPOINT_STEP ?=
PROMPT ?= i wake up.
PROFILE ?= dusty8m
CHAT_PROFILE ?=
TOP_P ?=
TEMPERATURE ?=
MAX_TOKENS ?=
MAX_CHAT_TURNS ?=
CHECKPOINT_PATH ?=
TOKENIZER_PATH ?=
DEVICE ?=
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
DUSTY_CHAT_REPO ?= mkhordoo/dusty-chat
DUSTY_CHAT_FILE ?= dusty_sft.jsonl
TINYSTORIES_SLICE ?= train[:100000]
TINYSTORIES_OUT ?= artifacts/datasets/tinystories_base.txt
ONNX_PROFILE ?= sft_dusty8m
ONNX_OUT ?= docs/model.onnx
ONNX_TOKENIZER_OUT ?= docs/tokenizer.json
WEB_PORT ?= 8000
HF_REPO_ID ?=
HF_PROFILE ?= sft_dusty8m
HF_STAGING_DIR ?= artifacts/hub_upload/$(HF_PROFILE)

.PHONY: help chat download-datasets dusty-generate-pretrain dusty-generate-sft dusty-filter-sft dusty-tokenizer dusty-pretrain-data dusty-pretrain dusty-generate dusty-sft-data dusty-sft-train serve-web dusty-export-onnx stage-hub push-hub tensorboard

help:
	@echo "DustyLM commands"
	@echo ""
	@echo "Dusty 8M workflow:"
	@echo "  make download-datasets          Download TinyStories + Dusty SFT data"
	@echo "  make dusty-generate-pretrain   Generate artifacts/datasets/dusty_pretrain.txt"
	@echo "  make dusty-generate-sft        Generate artifacts/datasets/dusty_sft.jsonl"
	@echo "  make dusty-filter-sft          Filter/sample SFT JSONL to artifacts/datasets/dusty_sft_2000.jsonl"
	@echo "  make dusty-tokenizer            Train artifacts/tokenizers/dusty_tokenizer.json"
	@echo "  make dusty-pretrain-data        Tokenize artifacts/datasets/dusty_pretrain.txt"
	@echo "  make dusty-pretrain EPOCHS=1    Train the dusty8m profile"
	@echo "  make dusty-sft-data             Tokenize artifacts/datasets/dusty_sft.jsonl"
	@echo "  make dusty-sft-train EPOCHS=1   Train the sft_dusty8m profile"
	@echo "  make chat                       Chat with the local SFT inference CLI"
	@echo "  make dusty-generate             Generate with PROFILE=dusty8m PROMPT='i wake up.'"
	@echo "  make serve-web                  Serve the browser demo locally"
	@echo "  make dusty-export-onnx          Export ONNX browser-demo artifacts to docs/"
	@echo "  make stage-hub HF_REPO_ID=mkhordoo/dusty-8m-sft"
	@echo "  make push-hub HF_REPO_ID=mkhordoo/dusty-8m-sft"
	@echo "  make tensorboard                Plot training logs from runs/"

download-datasets:
	uv run python data_pipeline/download_datasets.py \
		--tinystories-slice "$(TINYSTORIES_SLICE)" \
		--tinystories-out $(TINYSTORIES_OUT) \
		--dusty-pretrain-out $(DUSTY_PRETRAIN_OUT) \
		--dusty-chat-repo $(DUSTY_CHAT_REPO) \
		--dusty-chat-file $(DUSTY_CHAT_FILE) \
		--dusty-sft-out $(DUSTY_SFT_OUT)

dusty-generate-pretrain:
	uv run python data_pipeline/generate_pretrain.py \
		--model $(DUSTY_MODEL) \
		--workers $(DUSTY_PRETRAIN_WORKERS) \
		--out $(DUSTY_PRETRAIN_OUT) \
		--progress $(DUSTY_PRETRAIN_PROGRESS)

dusty-generate-sft:
	uv run python data_pipeline/generate_sft.py \
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
	uv run python data_pipeline/filter_sft.py \
		--input $(DUSTY_SFT_OUT) \
		--output $(DUSTY_SFT_FILTERED_OUT) \
		--target-total $(DUSTY_SFT_FILTER_TARGET) \
		--max-answer-tokens $(DUSTY_SFT_MAX_ANSWER_TOKENS) \
		--sampling-mode $(DUSTY_SFT_SAMPLING_MODE)

dusty-tokenizer:
	uv run python -m dustylm.tokenizer

dusty-pretrain-data:
	uv run python -m dustylm.data_prep --profile dusty8m

dusty-pretrain:
	uv run python -m dustylm.train --profile dusty8m --epochs $(EPOCHS) --checkpoint-every-steps $(CHECKPOINT_EVERY_STEPS)

dusty-generate:
	uv run python dustylm/generate.py --profile $(PROFILE) --prompt "$(PROMPT)" $(if $(TOP_P),--top-p $(TOP_P),) $(if $(TEMPERATURE),--temperature $(TEMPERATURE),) $(if $(CHECKPOINT_STEP),--checkpoint-step $(CHECKPOINT_STEP),)

chat:
	uv run python -m dustylm.inference $(if $(CHAT_PROFILE),--profile $(CHAT_PROFILE),)$(if $(CHECKPOINT_PATH), --checkpoint-path $(CHECKPOINT_PATH),)$(if $(TOKENIZER_PATH), --tokenizer-path $(TOKENIZER_PATH),)$(if $(DEVICE), --device $(DEVICE),)$(if $(TEMPERATURE), --temperature $(TEMPERATURE),)$(if $(MAX_TOKENS), --max-tokens $(MAX_TOKENS),)$(if $(TOP_P), --top-p $(TOP_P),)$(if $(MAX_CHAT_TURNS), --max-chat-turns $(MAX_CHAT_TURNS),)

dusty-sft-data:
	uv run python -m dustylm.data_prep --profile sft_dusty8m

dusty-sft-train:
	uv run python -m dustylm.train --profile sft_dusty8m --epochs $(EPOCHS) --checkpoint-every-steps $(CHECKPOINT_EVERY_STEPS)

dusty-export-onnx:
	uv run --extra onnx python scripts/export_onnx.py --profile $(ONNX_PROFILE) $(if $(CHECKPOINT_STEP),--checkpoint-step $(CHECKPOINT_STEP),) --output $(ONNX_OUT) --tokenizer-output $(ONNX_TOKENIZER_OUT)

serve-web:
	uv run --extra onnx python scripts/export_onnx.py --profile $(ONNX_PROFILE) $(if $(CHECKPOINT_STEP),--checkpoint-step $(CHECKPOINT_STEP),) --output $(ONNX_OUT) --tokenizer-output $(ONNX_TOKENIZER_OUT)
	@echo "Starting local web server..."
	@echo "Open http://localhost:$(WEB_PORT) in your browser to chat with DustyLM."
	uv run python -m http.server $(WEB_PORT) --directory docs

stage-hub:
ifndef HF_REPO_ID
	$(error HF_REPO_ID is undefined. Please run: make stage-hub HF_REPO_ID=your-username/repo-name)
endif
	@echo "Staging Hugging Face Hub artifacts locally for $(HF_REPO_ID) (dry run; no upload)."
	uv run --extra onnx --extra hub python scripts/push_to_hub.py \
		--repo-id $(HF_REPO_ID) \
		--profile $(HF_PROFILE) \
		--staging-dir $(HF_STAGING_DIR) \
		--dry-run

push-hub:
ifndef HF_REPO_ID
	$(error HF_REPO_ID is undefined. Please run: make push-hub HF_REPO_ID=your-username/repo-name)
endif
	@echo "Pushing staged DustyLM artifacts to Hugging Face Hub repo $(HF_REPO_ID)."
	uv run --extra onnx --extra hub python scripts/push_to_hub.py \
		--repo-id $(HF_REPO_ID) \
		--profile $(HF_PROFILE) \
		--staging-dir $(HF_STAGING_DIR)

tensorboard:
	uv run tensorboard --logdir runs
