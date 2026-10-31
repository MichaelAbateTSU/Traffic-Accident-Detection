"""Keep the configured CPU budget after Ultralytics initializes its predictor."""

from app.core.config import settings


def configure_production_model(model) -> None:
    import torch

    model.to("cpu")

    def enforce_cpu_threads(_predictor) -> None:
        # select_device resets this during first predictor setup.
        torch.set_num_threads(settings.inference_threads)

    model.add_callback("on_predict_start", enforce_cpu_threads)
