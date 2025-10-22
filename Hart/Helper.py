import tensorflow as tf
import pathlib

def Imfile(image, shape, scale=255.0, expand=True, dtype=tf.float32):
  """
  Helper to Open & Preprocess an Image File
  """
  if not isinstance(image, (str, pathlib.Path)):
    raise TypeError("Image Path Must be String or Pathlib.Path")
  image = tf.keras.utils.load_img(image, target_size=shape)
  image = tf.keras.utils.img_to_array(image)
  image = image / scale
  image = tf.cast(image, dtype)
  image = tf.expand_dims(image, axis=0) if expand else image
  return image

def Helite(modelfile, filename):
  """
  Helper to Convert TensorFlow Model to TFLite
  """
  filename = pathlib.Path(filename)

  if not pathlib.Path(modelfile).exists():
    # Unavailable Model File
    raise FileNotFoundError(f"Model Path '{modelfile}' Doesn't Exist !")
  # Convert to TensorFlow Lite Model
  converter = tf.lite.TFLiteConverter.from_saved_model(modelfile)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
  # Perform Conversion
  try:
    tflite_model = converter.convert()
  except Exception:
    raise RuntimeError("TensorFlow Lite Conversion Failed !")
  # Write to File
  filename.write_bytes(tflite_model)
  # Return
  return filename