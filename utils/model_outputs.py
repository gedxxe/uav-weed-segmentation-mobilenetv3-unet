import torch


SEGMENTATION_KEY = "segmentation"
FOREGROUND_KEY = "foreground"


def get_segmentation_logits(model_output):
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, dict) and SEGMENTATION_KEY in model_output:
        return model_output[SEGMENTATION_KEY]
    raise TypeError(
        "Model output must be a tensor or a dict containing "
        f"'{SEGMENTATION_KEY}' logits."
    )


def get_foreground_logits(model_output):
    if isinstance(model_output, dict) and FOREGROUND_KEY in model_output:
        return model_output[FOREGROUND_KEY]
    raise TypeError(
        "Auxiliary foreground loss requires model output dict containing "
        f"'{FOREGROUND_KEY}' logits."
    )
