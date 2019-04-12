# Overfit GPT model to "the quick brown fox"
#
# 906.45 -- the a , " he said . " i 'm not
# 310.08 -- the i - "   " i 'm not going to
# 134.41 -- the i - "   " i 'm not a child
# 30.41 -- the i - "   " i 'm not going to
#  8.07 -- the quick , " he said , " i 'm not
#  3.61 -- the quick quick quick steps , and then the quick quick
#  2.15 -- the quick quick quick jumps over the low fence jumps over
#  1.41 -- the quick fox jumps over the lazy dog jumps over the
#  1.13 -- the quick fox jumps over the lazy dog jumps over the
#  1.05 -- the quick quick brown fox jumps over the lazy dog jumps
#  1.02 -- the quick brown fox jumps over the lazy dog jumps over
#  1.01 -- the quick jumps over the lazy dog jumps over the lazy
#  1.02 -- the quick brown fox jumps over the lazy dog jumps over
#  1.13 -- the quick brown fox jumps over the lazy dog jumps over
#  1.02 -- the quick brown fox jumps over the lazy dog jumps over
#  1.00 -- the quick brown fox jumps over the lazy dog jumps over
#  1.01 -- the quick brown fox jumps over the lazy dog jumps over
#  1.00 -- the quick brown fox jumps over the lazy dog jumps over
#  1.00 -- the quick brown fox jumps over the lazy dog jumps over
#  1.00 -- the quick brown fox jumps over the lazy dog jumps over


import math
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

  start_words = ['the']
  start_tokens = [tokenizer.convert_tokens_to_ids(w) for w in start_words]
  
  for i in range(20):
    loss = model(input_ids=batch, lm_labels=batch)
    perplexity = math.exp(loss.item())
    print('%5.2f -- %s'%(perplexity, decode(start_tokens)))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    

if __name__=='__main__':
  main()
  
