
import tensorflow as tf

tf.data.experimental.CsvDataset(
    filenames, record_defaults, compression_type=None, buffer_size=None,
    header=False, field_delim=',', use_quote_delim=True,
    na_value='', select_cols=None, exclude_cols=None
)