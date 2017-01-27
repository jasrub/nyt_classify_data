#!/usr/bin/python
# -*- coding: utf-8 -*-
from magpie_mongo import MagpieModel
from keras.models import load_model
from magpie_mongo.utils import load_from_disk, save_to_disk
import gensim
import json

print "Loading pre-trained word to vec model"
word2vecmodel = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
print "model_loaded"
labels = []
with open('labels.json') as data_file:
  labels = json.load(data_file)
scaler = load_from_disk('./scalar/scalar_labels_long')
keras_model = load_model('./saved_models_labels_long/weights.00-0.00.hdf5')

model = MagpieModel(keras_model=keras_model, word2vec_model=word2vecmodel, scaler=scaler, labels=labels)
#save_to_disk('./scalar/embeddings', model.word2vec_model)
print "predicting..."

res = model.predict_from_text(u'''Looks like The Weeknd and Selena Gomez just took their relationship to the next level...they got their friends' stamp of approval!

The rumored couple stepped out in Hollywood on Wednesday night with a group of their famous friends—including French Montana and Jaden Smith—enjoying a fun-filled evening at Dave and Buster's to play games, take photos and enjoy each other's company.

Onlookers tell E! News they looked "smitten" as they left the arcade at 3 a.m. (!!!). In fact, they were photographed holding hands, unconcerned about showing some sweet PDA among the group as they got in the same car and headed home together.
Photos
Selena Gomez's Best Looks
"Selena looked so happy and Abel was very chill, very relaxed," the onlooker dished. "They held hands as they left and it was very warm, you can tell they are really enjoying each other's company...[Selena] seems very at ease with him and you can still see the stars in her eyes. They are clearly smitten!"

SelGo looked fashionable in an oversized denim jacket and wide-legged jeans, baring her midriff in a crop-top and wearing her hair half-up in a bun on her head. The Weeknd, on the other hand, opted for an all-black ensemble with a gold jacket and tennis shoes.

Several of The Weeknd's friends were also spotted hanging with the hot, new couple, wearing jackets and sweatshirts promoting the Canadian singer's latest album Starboy as well as his record company, XO Records.

French Montana took to Instagram to share a snapshot from the evening, rocking a fur coat and Timberland shoes while posing alongside The Weeknd and Smith, who donned an oversized sweatshirt with colorful pants.
Watch
Selena Gomez Spotted Kissing The Weeknd

"YOUNG LEGENDS #newboyband #shwag," the rapper captioned the pic.

Meanwhile, this is the second time we've seen "The Hills" singer and Gomez out together.

Earlier this month, they made headlines for their relationship after enjoying a long date night at the Santa Monica hot spot Giorgio Baldi during which they were photographed kissing and holding on to each other.

Now, it's been nearly two weeks since those pics surfaced, and a source recently told us the pair continue "hanging out" and growing closer.

"He really likes her. They text every day," our insider revealed. "They have a really sexy and flirty relationship. They laugh and like each other's personalities."

However, they're not in any rush to make things super serious. "As of now, they are just having fun and enjoying each other," our source said.
Watch
How Is Bella Hadid Doing Since Kissing Pics?

Another source has reiterated that point, telling us, "They are taking things slow and getting to know each other. Selena was focusing on getting herself back together and The Weeknd was just getting out of a relationship [with Bella Hadid], plus putting out his new album."

Still, the insider added, "He thinks she is extremely talented and sexy."

Speaking of Hadid, we're told the model is still working on moving forward from her ex-boyfriend.

"She is actually not over The Weeknd. She still loves him," another source revealed. "They are on fine terms, but she is bitter about the romance with Selena. She was not happy when all of that gossip went everywhere between The Weeknd and Selena."
Our insider added, "It really hurt her seeing Selena be all up on her man. She still feels like they have a connection."''')
print res [:20]