from magpie_mongo import MagpieModel
import magpie_mongo.utils
import json
import gensim
import os
import keras
import pymongo
import random

client = pymongo.MongoClient('mongodb://um.media.mit.edu:27017/super-glue')
db = client.nyt_corpus
mongo_collection = db.articles

last_id = mongo_collection.find({}).count()
all_ids = [str(i) for i in range(1, last_id)]
test_size = int(last_id*0.2)
random.shuffle(all_ids)

train_ids = all_ids[test_size:]
test_ids = all_ids[:test_size]

labels = []

with open('descriptors.json') as data_file:
  labels = [w["word"] for w in json.load(data_file)]

print "loaded ids and labels"
print "train %d  test: %d  labels: %d"%(len(train_ids), len(test_ids), len(labels))

# Load Google's pre-trained Word2Vec model.
print "Loading pre-trained word to vec model"
word2vecmodel = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
print "model_loaded"

# scaler = magpie_mongo.utils.load_from_disk('./scalar/scalar_all_labels')

model = MagpieModel(word2vec_model=word2vecmodel, labels=labels)

print "training for 5 epochs"
print (model)
model.fit_scaler_mongo(mongo_collection, train_ids)
magpie_mongo.utils.save_to_disk('./scalar/scalar_all_labels', model.scaler, overwrite=True)
save_path='./saved_models_all_labels'
if not os.path.exists(save_path):
    os.makedirs(save_path)
filepath = save_path+"/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
save_chackpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.batch_train_mongo(mongo_collection, train_ids, labels, test_ids=test_ids, nb_epochs=5, callbacks=[save_chackpoint])
magpie_mongo.utils.save_to_disk('./scalar/scalar_all_labels2', model.scaler, overwrite=True)

model.keras_model.save(os.path.join(save_path,'trained_model_all_labels.h5'))
