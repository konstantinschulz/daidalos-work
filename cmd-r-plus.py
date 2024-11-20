import os

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
model_id = "CohereForAI/c4ai-command-r-plus-4bit"
hf_token: str = os.getenv("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)

chunk = ["αὔριον",
         "αὔτως",
         "αὖθις",
         "αὗ",
         "βάθος",
         "βάθρον",
         "βάλανος",
         "βάλλω",
         "βάπτισμα",
         "βάπτω",
         "βάραθρον",
         "βάρβαρος",
         "βάρος",
         "βάσανος",
         "βάσις",
         "βάτος", ]
# Format message with the command-r-plus chat template
messages = [{"role": "user",
             "content": f"In the following, I will provide a list of Ancient Greek words to you. Add sentiment labels to each word according to the following schema: -1 (very negative), -0.5 (slightly negative), 0 (neutral), 0.5 (slightly positive), 1 (very positive). Use this output format 'GREEK_WORD: NUMERIC_LABEL', putting each entry on a new line, without index numbers, copying the GREEK_WORD exactly from the input. Do not add explanations or notes to the output. Here comes the list: \n ``` {chunk} ```"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
)

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
