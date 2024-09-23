import tensorflow as tf

# Path to your local .tfrecord file
file_path = r'C:\Users\Ezra\Documents\MIDI_DPG\Data\root\tensorflow_datasets\groove\full-midionly\2.0.1\groove-test.tfrecord-00000-of-00001'

# Define a function to parse the TFRecord data
def _parse_function(proto):
    # Define the feature description dictionary based on the inspection
    feature_description = {
        'midi': tf.io.FixedLenFeature([], tf.string),
        'style/primary': tf.io.FixedLenFeature([], tf.int64),
        'style/secondary': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'bpm': tf.io.FixedLenFeature([], tf.int64),
        'drummer': tf.io.FixedLenFeature([], tf.int64),
        'id': tf.io.FixedLenFeature([], tf.string),
        'time_signature': tf.io.FixedLenFeature([], tf.int64),
        'type': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(proto, feature_description)

# Load the TFRecord dataset from the file path
raw_dataset = tf.data.TFRecordDataset(file_path)

# Parse the dataset using the parsing function
dataset = raw_dataset.map(_parse_function)

# Build your input pipeline
dataset = dataset.shuffle(1024)

# Iterate through the dataset and access features
for features in dataset.take(1):
    midi = features['midi']
    primary_style = features['style/primary']
    secondary_style = features['style/secondary']
    bpm = features['bpm']
    drummer = features['drummer']
    id_ = features['id']
    
    print(f"MIDI: {midi}")
    print(f"Primary Style: {primary_style}")
    print(f"Secondary Style: {secondary_style}")
    print(f"BPM: {bpm}")
    print(f"Drummer: {drummer}")
    print(f"ID: {id_}")
