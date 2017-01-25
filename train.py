from magpie_mongo import MagpieModel
import magpie_mongo.utils
import json
import gensim
import os
import keras
import pymongo

client = pymongo.MongoClient('mongodb://um.media.mit.edu:27017/super-glue')
db = client.nyt_corpus
mongo_collection = db.articles

train_ids = []
test_ids = []
labels = []
with open('train_ids.json', 'r') as train_file:
    train_ids=json.load(train_file)

with open('test_ids.json', 'r') as test_file:
    test_ids = json.load(test_file)

with open('labels_long.json') as data_file:
  labels = json.load(data_file)

print "loaded ids and labels"
print "train %d  test: %d  labels: %d"%(len(train_ids), len(test_ids), len(labels))

# Load Google's pre-trained Word2Vec model.
print "Loading pre-trained word to vec model"
word2vecmodel = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
print "model_loaded"

#scaler = magpie_mongo.utils.load_from_disk('./scalar/scalar')

model = MagpieModel(word2vec_model=word2vecmodel, labels=labels)

print "training for 5 epochs"
print (model)
model.fit_scaler_mongo(mongo_collection, train_ids)
magpie_mongo.utils.save_to_disk('./scalar/scalar_labels_long', model.scaler, overwrite=True)
save_path='./saved_models_labels_long'
if not os.path.exists(save_path):
    os.makedirs(save_path)
filepath = save_path+"/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
save_chackpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.batch_train_mongo(mongo_collection, train_ids, labels, test_ids=test_ids, nb_epochs=5, callbacks=[save_chackpoint])
magpie_mongo.utils.save_to_disk('./scalar/scalar_labels_long2', model.scaler, overwrite=True)

model.keras_model.save(os.path.join(save_path,'trained_model_labels_long.h5'))
