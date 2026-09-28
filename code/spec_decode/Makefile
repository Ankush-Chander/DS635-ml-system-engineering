# Speculative decoding — Route A
#
#   make stage0     rejection sampling, no model      (cpu, seconds)
#   make stages     stages 1-4 against the default pair
#   make sweep      k sweep -> results/sweep.csv
#   make all        everything, in ladder order
#
#   make TARGET=Qwen/Qwen2.5-7B-Instruct sweep

PY     ?= python3
TARGET ?= Qwen/Qwen2.5-1.5B-Instruct
DRAFT  ?= Qwen/Qwen2.5-0.5B-Instruct
DEVICE ?= cuda
K      ?= 4

all: stage0 stages sweep

stage0:
	$(PY) stage0_rejection.py

stage1:
	$(PY) stage1_logits.py --model $(TARGET) --device $(DEVICE)

stage2:
	$(PY) stage2_kvcache.py --model $(TARGET) --device $(DEVICE)

stage3:
	$(PY) stage3_greedy_spec.py --target $(TARGET) --draft $(DRAFT) --drafter model --k $(K) --device $(DEVICE)

stage3-lookup:
	$(PY) stage3_greedy_spec.py --target $(TARGET) --drafter lookup --k $(K) --device $(DEVICE)

stage4:
	$(PY) stage4_sampled_spec.py --target $(TARGET) --draft $(DRAFT) --k $(K) --device $(DEVICE) --verify

stages: stage1 stage2 stage3-lookup stage3 stage4

sweep:
	$(PY) measure.py --target $(TARGET) --draft $(DRAFT) --device $(DEVICE)

clean:
	rm -rf results/*.csv __pycache__ common/__pycache__

.PHONY: all stage0 stage1 stage2 stage3 stage3-lookup stage4 stages sweep clean
