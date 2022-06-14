import re
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

CLEANR = re.compile('<.*?>') 

tokenizer = AutoTokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained("t5-base")
model =T5ForConditionalGeneration.from_pretrained('/home/joaquimneto04/trained_model/t5_base_webnlg/pytorch_model.bin', config=config)

def generate(text,model,tokenizer):
   model.eval()
   input_ids = tokenizer.encode(text, return_tensors="pt")
   outputs = model.generate(input_ids)
   return tokenizer.decode(outputs[0])


def cleanTags(raw_text):
  cleantext = re.sub(CLEANR, '', raw_text)
  return cleantext


def bleu(ref, gen):
	ref_bleu = []
	gen_bleu = []
	for l in gen:
		gen_bleu.append(l.split())
	for i,l in enumerate(ref):
		ref_bleu.append([l.split()])
	cc = SmoothingFunction()
	score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    
	return score_bleu

## Eval TEST with BLEU

df = pd.read_csv('/home/joaquimneto04/WebNLG/webNLG2020_test.csv')

generated_sentences = []

input_tuples = df['input_text'].values.tolist()
reference_sentences = df['target_text'].values.tolist()

for input_t, ref_sent in zip(input_tuples, reference_sentences):
	generated_sent = generate(input_t, model, tokenizer)
	generated_sent = cleanTags(generated_sent)
	generated_sent = generated_sent.strip()
	generated_sentences.append(generated_sent)


print('General BLEU Metric')
print(' ')

print(bleu(reference_sentences, generated_sentences))


## Show BLEU > .7

print(' ')
print('All BLEU Metrics > .7')
print(' ')

bleuMetrics = []

cc = SmoothingFunction()

for r, g in zip(reference_sentences, generated_sentences):
	m = sentence_bleu(r, g, smoothing_function=cc.method4)
	bleuMetrics.append(m)

for i in enumerate(bleuMetrics):
	if i[1] > 0.7:
		print(i)


## Show all BLEU Metrics

print(' ')
print('All BLEU Metrics')
print(' ')

for i in enumerate(bleuMetrics):
	print(i)


## TEST Sentences

print(' ')
print('TEST Sentences (BASELINE)')
print(' ')

print(input_tuples[358])
print(reference_sentences[358])
print(generated_sentences[358])
sentence_bleu(reference_sentences[358], generated_sentences[358], smoothing_function=cc.method4)

print(' ')

print(input_tuples[371])
print(reference_sentences[371])
print(generated_sentences[371])
sentence_bleu(reference_sentences[371], generated_sentences[371], smoothing_function=cc.method4)

print(' ')

print(input_tuples[466])
print(reference_sentences[466])
print(generated_sentences[466])
sentence_bleu(reference_sentences[466], generated_sentences[466], smoothing_function=cc.method4)

print(' ')

print(input_tuples[838])
print(reference_sentences[838])
print(generated_sentences[838])
sentence_bleu(reference_sentences[838], generated_sentences[838], smoothing_function=cc.method4)

print(' ')

print(input_tuples[1622])
print(reference_sentences[1622])
print(generated_sentences[1622])
sentence_bleu(reference_sentences[1622], generated_sentences[1622], smoothing_function=cc.method4)


## Ontology generating sentences

print(' ')
print('Ontology Sentences (BASELINE)')
print(' ')

print(generate('alveolus |part_of| alveolar_system | part_of | lung | part_of | respiratory_system', model, tokenizer))

print(generate('bowel | part_of | gastrointestinal_system | part_of |digestive_system', model, tokenizer))

print(generate('chest | part_of | thorax | part_of | trunk', model, tokenizer))

print(generate('cranium | part_of | head_bone', model, tokenizer))

print(generate('epithalamus | part_of | diencephalon | part_of | forebrain | part_of | brain', model, tokenizer))

print(generate('knee | part_of | hindlimb', model, tokenizer))

print(generate('lens | part_of | eye_anterior_segment | part_of | eye', model, tokenizer))

print(generate('meninges | part_of | central_nervous_system | part_of | nervous_system', model, tokenizer))

print(generate('neck | part_of | head', model, tokenizer))

print(generate('tarsus | part_of | foot', model, tokenizer))