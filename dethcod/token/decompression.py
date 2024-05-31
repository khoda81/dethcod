import transformers
import transformers.modeling_outputs


class DecompressionConfig(transformers.T5Config): ...


class DecompressionModel(transformers.T5ForConditionalGeneration): ...
