import transformers
import transformers.modeling_outputs


class CompressionConfig(transformers.T5Config): ...


class CompressionModel(transformers.T5ForConditionalGeneration): ...
