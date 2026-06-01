from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "./"
)

model.save_pretrained(
    "./sharded",
    safe_serialization=True,
    max_shard_size="150MB"
)
