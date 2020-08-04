from langid.langid import LanguageIdentifier, model

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
lang, prob = identifier.classify("ottimo rapporto qualitprezzo , al di sopra delle aspettative personale cordiale e disponibile ottimi servizi consigliato")
# res = identifier.classify("I have been living in the UK for about 10 years now.")
print(lang)
print(prob)


for i in range(10):
	print(i)
	continue