import os  # used for directory operations
import tensorflow as tf
from PIL import Image  # used to read images from directory

# my file path, just like the picture above
cwd = "/path_to_image_folder"
# the tfrecord file path, you need to create the folder yourself
recordPath =  "path_to_output_folder/tfrecord_test/"
# the best number of images stored in each tfrecord file
bestNum = 1000
# the index of images flowing into each tfrecord file
num = 0
# the index of the tfrecord file
recordFileNum = 0
# the number of classes of images
keys = ["No_Fire", "Fire"]
values = [i for i in list(range(2))]
classes = dict(zip(keys, values))
# name format of the tfrecord files
recordFileName = ("test-%.3d.tfrec" % recordFileNum)
# tfrecord file writer
writer = tf.io.TFRecordWriter(recordPath + recordFileName)

print("Creating the 000 tfrecord file")
for name, label in classes.items():
    print(name)
    print(label)
    class_path = os.path.join(cwd, name)
    for img_name in os.listdir(class_path):
       
        recordFileNum += 1
        name_of_file = ("test-%.3d.tfrec" % recordFileNum)
        writer = tf.io.TFRecordWriter(recordPath + name_of_file)
        print("Creating the %.3d tfrecord file" % recordFileNum)
        img_path = os.path.join(class_path, img_name)
        
        img_raw = open(img_path, 'rb').read()
        example = tf.train.Example(features=tf.train.Features(feature={
"img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
        writer.write(example.SerializeToString())
writer.close()