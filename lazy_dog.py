# Overfit GPT model to "the quick brown fox"
#
# 6.97 -- the quick brown fox fire .   " i 'm sorry , " he
# 6.08 -- the quick brown fox back .   " i 'm sorry , " he
# 4.39 -- the quick brown fox fox fox fox fox fox fox fox fox fox fox
# 3.60 -- the quick brown fox .   " i 'm sorry , " he said
# 2.36 -- the quick brown fox fox fox fox tail   i do n't know what
# 1.31 -- the quick brown fox fox   " i 'm not sure what to say
# 0.85 -- the quick brown fox fox fox quick .   " i 'm not sure
# 0.42 -- the quick brown fox fox jumps over the fox jumps over the fox jumps
# 0.18 -- the quick brown fox jumps over the lazy dog jumps over the lazy dog
# 0.16 -- the quick brown fox quick quick quick quick brown fox jumps over the lazy
# 0.03 -- the quick brown fox quick brown fox jumps over the lazy dog jumps over
# 0.02 -- the quick brown fox quick jumps over the lazy dog jumps over the lazy
# 0.03 -- the quick brown fox quick jumps over the lazy dog jumps over the lazy
# 0.02 -- the quick brown fox quick brown fox jumps over the lazy dog jumps over
# 0.06 -- the quick brown fox jumps over the lazy dog jumps over the lazy dog


import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel


def argmax(t):
    return int(torch.argmax(t).detach().numpy())
  
def decode(start_tokens, length=10):
  result = []
  context = torch.ones(1, 0, dtype=torch.long)
  for start_token in start_tokens:
    new_token = torch.full((1, 1), start_token, dtype=torch.long)
    context = torch.cat((context, new_token), dim=1)
    result.append(tokenizer.convert_ids_to_tokens([start_token])[0])

  with torch.no_grad():
    for i in range(length):
      logits = model(context)  # batch_size x 1
      predicted_id = argmax(logits[0,-1])
      predicted_word = tokenizer.convert_ids_to_tokens([predicted_id])[0]
      tokenizer.convert_ids_to_tokens([])
      if predicted_word.endswith('</w>'):
        predicted_word = predicted_word[:-len('</w>')]
      result.append(predicted_word)

      predicted_id_batch = torch.tensor([[predicted_id]])
      context = torch.cat((context, predicted_id_batch), dim=1)

  result = ' '.join(result)
  result = result.replace('\n', ' ')
  return result


def main():
  global tokenizer, model

  train_dataset = 'the quick brown fox jumps over the lazy dog'
  tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
  tokenized = [tokenizer.tokenize(train_dataset)]

  # [[481, 2279, 2507, 8573, 11670, 715, 481, 8447, 2585]]
  encoded = [tokenizer.convert_tokens_to_ids(t) for t in tokenized]
  model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

  optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

  
  batch = torch.tensor(encoded)

  start_words = ['the', 'quick', 'brown', 'fox']
  start_tokens = [tokenizer.convert_tokens_to_ids(w) for w in start_words]
  
  for i in range(40):
    loss = model(input_ids=batch, lm_labels=batch)
    print('%.2f -- %s'%(loss.detach().numpy(), decode(start_tokens)))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    

if __name__=='__main__':
  main()
  
