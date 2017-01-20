from magpie import MagpieModel
import magpie.utils
import json
import gensim

# Load Google's pre-trained Word2Vec model.
print "Loading pre-trained word to vec model"
word2vecmodel = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
print "model_loaded"
labels = []
with open('labels.json') as data_file:
  labels = json.load(data_file)
scaler = magpie.utils.load_from_disk('./scalar/scalar')
print ("%d labels"%len(labels))
print labels[:20]
train_dir = "./train"
model = MagpieModel(word2vec_model=word2vecmodel, labels=labels, scaler=scaler)
print "initializing word vectors..."

#magpie.utils.save_to_disk('embeddings', model.word2vec_model)
# magpie.utils.save_to_disk('./scalar/scalar', model.scaler, overwrite=True)

print "embeddings saved to disk"
print "training for 5 epochs"
print (model)
# model.fit_scaler('./train')
# magpie.utils.save_to_disk('./scalar/scalar', model.scaler, overwrite=True)
model.batch_train(train_dir, word2vecmodel.vocab.keys(), test_dir='./test', nb_epochs=5)
magpie.utils.save_to_disk('./scalar/scalar', model.scaler, overwrite=True)

model.keras_model.save('trained_model.h5')
