CYAN   := \033[0;36m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
BOLD   := \033[1m
DIM    := \033[2m
NC     := \033[0m

EPOCHS ?= 23
CHECKPOINT_EVERY_STEPS ?= 100
CHECKPOINT_STEP ?=
PROMPT ?= $(strip Once upon a time)
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
BATCH_SIZE ?= 32
DUSTY_SFT_BATCH_SIZE ?= 20
DUSTY_SFT_OUT ?= artifacts/datasets/dusty_sft.jsonl
DUSTY_SFT_REJECTED ?= artifacts/datasets/dusty_sft_rejected.jsonl
DUSTY_SFT_FILTERED_OUT ?= artifacts/datasets/dusty_sft_2000.jsonl
DUSTY_SFT_FILTER_TARGET ?= 2000
DUSTY_SFT_MAX_ANSWER_TOKENS ?= 256
DUSTY_SFT_SAMPLING_MODE ?= balanced
DUSTY_CHAT_REPO ?= mkhordoo/dusty-chat
DUSTY_CHAT_FILE ?= dusty_sft.jsonl
TINYSTORIES_SLICE ?= train[:50000]
TINYSTORIES_OUT ?= artifacts/datasets/tinystories_base.txt
ONNX_PROFILE ?= sft_dusty8m
ONNX_OUT ?= docs/model.onnx
ONNX_TOKENIZER_OUT ?= docs/tokenizer.json
WEB_PORT ?= 8000
HF_REPO_ID ?=
HF_PROFILE ?= sft_dusty8m
HF_STAGING_DIR ?= artifacts/hub_upload/$(HF_PROFILE)
HF_SKIP_ONNX ?=
HF_SKIP_CHAT_TEMPLATE ?=
HF_BASE_REPO_ID ?= mkhordoo/dusty-8m-base
HF_SFT_REPO_ID ?= mkhordoo/dusty-8m-sft
HF_BASE_PROFILE ?= dusty8m
HF_SFT_PROFILE ?= sft_dusty8m
HF_BASE_MODEL_CARD ?= artifacts/hf/HF_BASE_MODEL_CARD.md
HF_SFT_MODEL_CARD ?= artifacts/hf/HF_MODEL_CARD.md
HUB_TARGET ?= all
EVAL_PROFILE ?= sft_dusty8m
EVAL_INPUT_SET ?= auto
EVAL_INPUTS ?=
EVAL_STEPS ?=
EVAL_OUTPUT_DIR ?= artifacts/evaluations/checkpoints
EVAL_RUN_ID ?=
EVAL_TOP_P ?=
EVAL_TEMPERATURE ?=
EVAL_MAX_NEW_TOKENS ?=

.PHONY: help chat download-models download-datasets synthesize-sft filter-sft tokenizer data-pretrain train-pretrain generate data-sft train-sft eval-checkpoints serve-web export-onnx stage-hub push-hub tensorboard train-end-to-end

help:
	@printf "$(BOLD)$(CYAN)DustyLM commands$(NC)\n"
	@printf "\n"
	@printf "$(BOLD)Quick Start (Inference Path):$(NC)\n"
	@printf "  make download-models            Download pretrained weights + tokenizer from Hugging Face Hub\n"
	@printf "  make chat                       Chat with local SFT inference CLI\n"
	@printf "  make generate                   Generate text with the dusty8m profile\n"
	@printf "\n"
	@printf "$(BOLD)Data:$(NC)\n"
	@printf "  make download-datasets          Download raw training datasets (TinyStories + SFT logs)\n"
	@printf "  make synthesize-sft             Synthesize raw SFT chat data via LLM\n"
	@printf "  make filter-sft                 Filter/sample SFT JSONL\n"
	@printf "\n"
	@printf "$(BOLD)Tokenizer & Datasets:$(NC)\n"
	@printf "  make tokenizer                  Train tokenizer\n"
	@printf "  make data-pretrain              Tokenize pretrain text\n"
	@printf "  make data-sft                   Tokenize SFT chat data\n"
	@printf "\n"
	@printf "$(BOLD)Training:$(NC)\n"
	@printf "  make train-pretrain EPOCHS=1    Train the dusty8m profile\n"
	@printf "  make train-sft EPOCHS=1         Train the sft_dusty8m profile\n"
	@printf "\n"
	@printf "$(BOLD)Evaluation:$(NC)\n"
	@printf "  make eval-checkpoints EVAL_STEPS=\"900 19600\"  Compare checkpoint outputs\n"
	@printf "\n"
	@printf "$(BOLD)Export:$(NC)\n"
	@printf "  make serve-web                  Serve browser demo locally\n"
	@printf "  make export-onnx                Export ONNX artifacts to docs/\n"
	@printf "\n"
	@printf "$(BOLD)Hub:$(NC)\n"
	@printf "  make stage-hub                  Stage both base + SFT artifacts locally (dry run)\n"
	@printf "  make stage-hub HUB_TARGET=base  Stage only base artifacts\n"
	@printf "  make stage-hub HUB_TARGET=sft   Stage only SFT artifacts\n"
	@printf "  make stage-hub HF_REPO_ID=...   Stage a single repo (override profile with HF_PROFILE=)\n"
	@printf "  make push-hub                   Push both base + SFT to Hugging Face Hub\n"
	@printf "  make push-hub HUB_TARGET=base   Push only base repo\n"
	@printf "  make push-hub HUB_TARGET=sft    Push only SFT repo\n"
	@printf "  make push-hub HF_REPO_ID=...    Push a single repo\n"
	@printf "  make tensorboard                Plot training logs from artifacts/tensorboard/\n"

# =============================================================================
# 1. Model & Artifacts Pipeline
# =============================================================================

download-models:
	@mkdir -p artifacts/checkpoints artifacts/tokenizers
	@printf "$(CYAN)Downloading Dusty 8M Base (pre-trained weights)...$(NC)\n"
	hf download mkhordoo/dusty-8m-base model.pt --local-dir artifacts/checkpoints > /dev/null 2>&1
	@mv -f artifacts/checkpoints/model.pt artifacts/checkpoints/dusty8m.pt
	@printf "$(GREEN)  ✔ dusty8m.pt$(NC)\n"
	@printf "$(CYAN)Downloading Dusty 8M SFT (vacuum persona)...$(NC)\n"
	hf download mkhordoo/dusty-8m-sft model.pt --local-dir artifacts/checkpoints > /dev/null 2>&1
	@mv -f artifacts/checkpoints/model.pt artifacts/checkpoints/dusty8m_sft.pt
	@printf "$(GREEN)  ✔ dusty8m_sft.pt$(NC)\n"
	@printf "$(CYAN)Downloading Dusty tokenizer...$(NC)\n"
	hf download mkhordoo/dusty-8m-base tokenizer.json --local-dir artifacts/tokenizers > /dev/null 2>&1
	@mv -f artifacts/tokenizers/tokenizer.json artifacts/tokenizers/dusty_tokenizer.json
	@printf "$(GREEN)  ✔ dusty_tokenizer.json$(NC)\n"
	@printf "$(GREEN)✔ Models ready. Run 'make chat' or 'make generate'.$(NC)\n"

# =============================================================================
# 2. Data Engineering Pipeline
# =============================================================================

# Defaults to an optimized 100k slice of TinyStories — enough to train the 8M
# model to solid English grammar on standard hardware. To experiment with more
# data on a high-end GPU, override the slice:
#   make download-datasets TINYSTORIES_SLICE="train"
#   make download-datasets TINYSTORIES_SLICE="train[:2000000]"

download-datasets:
	@printf "$(YELLOW)Downloading raw training datasets (TinyStories + Dusty SFT Conversations)...$(NC)\n"
	uv run python data_pipeline/download_datasets.py \
		--tinystories-slice "$(TINYSTORIES_SLICE)" \
		--tinystories-out $(TINYSTORIES_OUT) \
		--dusty-chat-repo $(DUSTY_CHAT_REPO) \
		--dusty-chat-file $(DUSTY_CHAT_FILE) \
		--dusty-sft-out $(DUSTY_SFT_OUT)
	@printf "$(GREEN)✔ Datasets ready for training. Run 'make train-pretrain' or 'make train-sft'.$(NC)\n"

synthesize-sft:
	@printf "$(YELLOW)Synthesizing SFT chat data via LLM...$(NC)\n"
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
	@printf "$(GREEN)✔ SFT chat data synthesized!$(NC)\n"

filter-sft:
	@printf "$(YELLOW)Filtering and sampling SFT dataset...$(NC)\n"
	uv run python data_pipeline/filter_sft.py \
		--input $(DUSTY_SFT_OUT) \
		--output $(DUSTY_SFT_FILTERED_OUT) \
		--target-total $(DUSTY_SFT_FILTER_TARGET) \
		--max-answer-tokens $(DUSTY_SFT_MAX_ANSWER_TOKENS) \
		--sampling-mode $(DUSTY_SFT_SAMPLING_MODE)
	@printf "$(GREEN)✔ SFT dataset filtered!$(NC)\n"

tokenizer:
	@printf "$(YELLOW)Training tokenizer...$(NC)\n"
	uv run python -m dustylm.tokenizer
	@printf "$(GREEN)✔ Tokenizer trained successfully!$(NC)\n"

data-pretrain:
	@printf "$(YELLOW)Clearing any existing pretrain tokenized data...$(NC)\n"
	@rm -rf artifacts/datasets/dusty_pretrain_tokenized
	@printf "$(YELLOW)Tokenizing pretrain data for dusty8m...$(NC)\n"
	uv run python -m dustylm.data_prep --profile dusty8m
	@printf "$(GREEN)✔ Pretrain data tokenized!$(NC)\n"

train-pretrain:
	@printf "$(YELLOW)Starting pretraining (dusty8m, $(EPOCHS) epochs)...$(NC)\n"
	uv run python -m dustylm.train --profile dusty8m --epochs $(EPOCHS) --checkpoint-every-steps $(CHECKPOINT_EVERY_STEPS) --batch-size $(BATCH_SIZE)
	@printf "$(GREEN)✔ Pretraining complete! Checkpoints saved.$(NC)\n"

generate:
	@printf "$(YELLOW)Generating text with $(PROFILE) profile...$(NC)\n"
	@uv run python scripts/check_artifacts.py $(PROFILE) generate $(if $(CHECKPOINT_STEP),--checkpoint-step $(CHECKPOINT_STEP),) || exit 1
	uv run python dustylm/generate.py --profile $(PROFILE) --prompt "$(strip $(PROMPT))" $(if $(TOP_P),--top-p $(TOP_P),) $(if $(TEMPERATURE),--temperature $(TEMPERATURE),) $(if $(CHECKPOINT_STEP),--checkpoint-step $(CHECKPOINT_STEP),)
	@printf "$(GREEN)✔ Generation complete!$(NC)\n"

chat:
	@printf "\n$(BOLD)$(CYAN)==================================================$(NC)\n"
	@printf "$(BOLD)$(CYAN)  DustyLM Live Interface$(NC)\n"
	@printf "$(BOLD)$(CYAN)==================================================$(NC)\n\n"
	@uv run python scripts/check_artifacts.py "$(CHAT_PROFILE)" chat || exit 1
	uv run python -m dustylm.inference $(if $(CHAT_PROFILE),--profile $(CHAT_PROFILE),)$(if $(CHECKPOINT_PATH), --checkpoint-path $(CHECKPOINT_PATH),)$(if $(TOKENIZER_PATH), --tokenizer-path $(TOKENIZER_PATH),)$(if $(DEVICE), --device $(DEVICE),)$(if $(TEMPERATURE), --temperature $(TEMPERATURE),)$(if $(MAX_TOKENS), --max-tokens $(MAX_TOKENS),)$(if $(TOP_P), --top-p $(TOP_P),)$(if $(MAX_CHAT_TURNS), --max-chat-turns $(MAX_CHAT_TURNS),)

data-sft:
	@printf "$(YELLOW)Clearing any existing SFT tokenized data...$(NC)\n"
	@rm -rf artifacts/datasets/dusty_sft_tokenized
	@printf "$(YELLOW)Tokenizing SFT data for sft_dusty8m...$(NC)\n"
	uv run python -m dustylm.data_prep --profile sft_dusty8m
	@printf "$(GREEN)✔ SFT data tokenized!$(NC)\n"

train-sft:
	@printf "$(YELLOW)Starting SFT fine-tuning (sft_dusty8m, $(EPOCHS) epochs)...$(NC)\n"
	uv run python -m dustylm.train --profile sft_dusty8m --epochs $(EPOCHS) --checkpoint-every-steps $(CHECKPOINT_EVERY_STEPS) --batch-size $(BATCH_SIZE)
	@printf "$(GREEN)✔ SFT fine-tuning complete! Checkpoint saved.$(NC)\n"

eval-checkpoints:
	@if [ -z "$(strip $(EVAL_STEPS))" ]; then \
		printf "$(RED)Set EVAL_STEPS, for example: make eval-checkpoints EVAL_STEPS=\"900 19600\"$(NC)\n"; \
		exit 1; \
	fi
	@printf "$(YELLOW)Comparing checkpoints for $(EVAL_PROFILE)...$(NC)\n"
	uv run python evaluation/compare_checkpoints.py \
		--profile $(EVAL_PROFILE) \
		--input-set $(EVAL_INPUT_SET) \
		--steps $(EVAL_STEPS) \
		--output-dir $(EVAL_OUTPUT_DIR) \
		$(if $(EVAL_INPUTS),--inputs $(EVAL_INPUTS),) \
		$(if $(EVAL_RUN_ID),--run-id $(EVAL_RUN_ID),) \
		$(if $(EVAL_TOP_P),--top-p $(EVAL_TOP_P),) \
		$(if $(EVAL_TEMPERATURE),--temperature $(EVAL_TEMPERATURE),) \
		$(if $(EVAL_MAX_NEW_TOKENS),--max-new-tokens $(EVAL_MAX_NEW_TOKENS),)
	@printf "$(GREEN)✔ Checkpoint comparison complete! Reports saved under $(EVAL_OUTPUT_DIR).$(NC)\n"

export-onnx:
	@printf "$(YELLOW)Exporting ONNX model...$(NC)\n"
	uv run --extra onnx python scripts/export_onnx.py --profile $(ONNX_PROFILE) $(if $(CHECKPOINT_STEP),--checkpoint-step $(CHECKPOINT_STEP),) --output $(ONNX_OUT) --tokenizer-output $(ONNX_TOKENIZER_OUT)
	@printf "$(GREEN)✔ ONNX artifacts exported!$(NC)\n"

serve-web:
	@printf "$(YELLOW)Exporting ONNX model and starting web server...$(NC)\n"
	uv run --extra onnx python scripts/export_onnx.py --profile $(ONNX_PROFILE) $(if $(CHECKPOINT_STEP),--checkpoint-step $(CHECKPOINT_STEP),) --output $(ONNX_OUT) --tokenizer-output $(ONNX_TOKENIZER_OUT)
	@printf "$(GREEN)✔ ONNX artifacts exported!$(NC)\n"
	@printf "$(YELLOW)Starting local web server...$(NC)\n"
	@printf "$(CYAN)Open http://localhost:$(WEB_PORT) in your browser to chat with DustyLM.$(NC)\n"
	uv run python -m http.server $(WEB_PORT) --directory docs

stage-hub:
ifdef HF_REPO_ID
	@printf "$(YELLOW)Staging artifacts locally for $(HF_REPO_ID) (dry run)...$(NC)\n"
	uv run --extra onnx --extra hub python scripts/push_to_hub.py \
		--repo-id $(HF_REPO_ID) \
		--profile $(HF_PROFILE) \
		--model-card $(HF_SFT_MODEL_CARD) \
		--staging-dir $(HF_STAGING_DIR) \
		$(HF_SKIP_ONNX) \
		$(HF_SKIP_CHAT_TEMPLATE) \
		--dry-run
	@printf "$(GREEN)✔ Staging complete. Review artifacts before running make push-hub.$(NC)\n"
else
	ifneq (,$(filter all base,$(HUB_TARGET)))
		@printf "$(YELLOW)Staging Dusty 8M Base ($(HF_BASE_REPO_ID))...$(NC)\n"
		$(MAKE) stage-hub \
			HF_REPO_ID=$(HF_BASE_REPO_ID) \
			HF_PROFILE=$(HF_BASE_PROFILE) \
			HF_SFT_MODEL_CARD=$(HF_BASE_MODEL_CARD) \
			HF_STAGING_DIR=artifacts/hub_upload/$(HF_BASE_PROFILE) \
			HF_SKIP_ONNX=--skip-onnx \
			HF_SKIP_CHAT_TEMPLATE=--skip-chat-template
	endif
	ifneq (,$(filter all sft,$(HUB_TARGET)))
		@printf "$(YELLOW)Staging Dusty 8M SFT ($(HF_SFT_REPO_ID))...$(NC)\n"
		$(MAKE) stage-hub \
			HF_REPO_ID=$(HF_SFT_REPO_ID) \
			HF_PROFILE=$(HF_SFT_PROFILE) \
			HF_STAGING_DIR=artifacts/hub_upload/$(HF_SFT_PROFILE) \
			HF_SKIP_ONNX= \
			HF_SKIP_CHAT_TEMPLATE=
	endif
	@printf "$(GREEN)✔ Staging complete. Review artifacts in artifacts/hub_upload/ before running make push-hub.$(NC)\n"
endif

push-hub:
ifdef HF_REPO_ID
	@printf "$(YELLOW)Pushing artifacts to Hugging Face Hub repo $(HF_REPO_ID)...$(NC)\n"
	uv run --extra onnx --extra hub python scripts/push_to_hub.py \
		--repo-id $(HF_REPO_ID) \
		--profile $(HF_PROFILE) \
		--model-card $(HF_SFT_MODEL_CARD) \
		--staging-dir $(HF_STAGING_DIR) \
		$(HF_SKIP_ONNX) \
		$(HF_SKIP_CHAT_TEMPLATE)
	@printf "$(GREEN)✔ Push to Hugging Face Hub complete!$(NC)\n"
else
	ifneq (,$(filter all base,$(HUB_TARGET)))
		@printf "$(YELLOW)Pushing Dusty 8M Base ($(HF_BASE_REPO_ID))...$(NC)\n"
		$(MAKE) push-hub \
			HF_REPO_ID=$(HF_BASE_REPO_ID) \
			HF_PROFILE=$(HF_BASE_PROFILE) \
			HF_SFT_MODEL_CARD=$(HF_BASE_MODEL_CARD) \
			HF_STAGING_DIR=artifacts/hub_upload/$(HF_BASE_PROFILE) \
			HF_SKIP_ONNX=--skip-onnx \
			HF_SKIP_CHAT_TEMPLATE=--skip-chat-template
	endif
	ifneq (,$(filter all sft,$(HUB_TARGET)))
		@printf "$(YELLOW)Pushing Dusty 8M SFT ($(HF_SFT_REPO_ID))...$(NC)\n"
		$(MAKE) push-hub \
			HF_REPO_ID=$(HF_SFT_REPO_ID) \
			HF_PROFILE=$(HF_SFT_PROFILE) \
			HF_STAGING_DIR=artifacts/hub_upload/$(HF_SFT_PROFILE) \
			HF_SKIP_ONNX= \
			HF_SKIP_CHAT_TEMPLATE=
	endif
	@printf "$(GREEN)✔ Hub push complete!$(NC)\n"
endif

tensorboard:
	@printf "$(YELLOW)Starting TensorBoard...$(NC)\n"
	uv run tensorboard --logdir artifacts/tensorboard
	@printf "$(GREEN)✔ TensorBoard stopped.$(NC)\n"

# =============================================================================
#  Automation Orchestration
# =============================================================================

train-end-to-end:
	@printf "$(CYAN)==========================================$(NC)\n"
	@printf "$(CYAN)   Starting End-to-End DustyLM Pipeline   $(NC)\n"
	@printf "$(CYAN)==========================================$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)[1/4] Tokenizing pre-training data...$(NC)\n"
	make data-pretrain
	@printf "\n"
	@printf "$(YELLOW)[2/4] Running pre-training phase (Epochs: 1)...$(NC)\n"
	make train-pretrain EPOCHS=1
	@printf "\n"
	@printf "$(YELLOW)[3/4] Generating SFT data...$(NC)\n"
	make data-sft
	@printf "\n"
	@printf "$(YELLOW)[4/4] Running SFT phase (Epochs: 21)...$(NC)\n"
	make train-sft EPOCHS=21
	@printf "\n"
	@printf "$(GREEN)✔ End-to-End Pipeline complete! All checkpoints saved to artifacts/.$(NC)\n"
