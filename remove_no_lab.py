import os
train_dir = './train'
test_dir = './test'
train_filenames = list({filename[:-4] for filename in os.listdir(train_dir) if filename.endswith('.txt')})
test_filenames = list({filename[:-4] for filename in os.listdir(test_dir) if filename.endswith('.txt')})

for f in train_filenames:
  filename = os.path.join(train_dir, f + '.lab')
  if not os.path.exists(filename):
        print f
        os.remove(os.path.join(train_dir, f + '.txt'))
for f in test_filenames:
  filename = os.path.join(test_dir, f + '.lab')
  if not os.path.exists(filename):
        print f
        os.remove(os.path.join(test_dir, f + '.txt'))
