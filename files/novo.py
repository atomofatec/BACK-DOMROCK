from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed

set_seed(52)
model = GPT2LMHeadModel.from_pretrained("egonrp/gpt2-medium-wikiwriter-squadv11-portuguese")
tokenizer = GPT2Tokenizer.from_pretrained("egonrp/gpt2-medium-wikiwriter-squadv11-portuguese")
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model.config.pad_token_id = tokenizer.eos_token_id

prompt_text = 'Aqui estão alguns comentários dos usuários referentes ao produto Copo Acrílico Com Canudo 500ml Rocie: por apenas R$1994,20 eu consegui comprar esse lindo copo de acrílico ótimo produto fácil instalação e muito bonita moderna muito bom a cada ano a jogabilidade fica mais parecido com a real muito bom produto excelenteentrega muito rápido recomendadíssimo ótimo produto original recomendo a todos podem comprar prazo de entrega dentro do esperado preço muito atrativo parabéns as americanas farei mais compras futuras. Por favor, gere dois resumos claro e conciso dessas avaliações, destacando os pontos positivos e negativos sobre o Copo Acrílico Com Canudo 500ml Rocie'

encoded_prompt = tokenizer.encode(prompt_text, return_tensors="pt")
output_sequences = model.generate(
    input_ids=encoded_prompt,
    do_sample=False,
    num_return_sequences=1,
    max_new_tokens=145,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.eos_token_id
)

decoded_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(decoded_text)