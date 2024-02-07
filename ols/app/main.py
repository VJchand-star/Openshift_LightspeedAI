"""Entry point to FastAPI-based web service."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI

from ols.app import metrics
from ols.app.endpoints import feedback, health, ols
from ols.src.ui.gradio_ui import GradioUI
from ols.utils import config
from ols.utils.logging import configure_logging

app = FastAPI(
    title="Swagger OpenShift LightSpeed Service - OpenAPI",
    description="""OpenShift LightSpeed Service API specification.""",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


cfg_file = os.environ.get("OLS_CONFIG_FILE", "olsconfig.yaml")
config.init_config(cfg_file)

configure_logging(config.ols_config.logging_config)
logger = logging.getLogger(__name__)
logger.info(f"Config loaded from {Path(cfg_file).resolve()}")


if config.dev_config.enable_dev_ui:
    app = GradioUI().mount_ui(app)
else:
    logger.info(
        "Embedded Gradio UI is disabled. To enable set enable_dev_ui: true "
        "in the dev section of the configuration file"
    )


def include_routers(app: FastAPI):
    """Include FastAPI routers for different endpoints.

    Args:
        app: The `FastAPI` app instance.
    """
    app.include_router(ols.router, prefix="/v1")
    app.include_router(feedback.router, prefix="/v1")
    app.include_router(health.router)
    app.mount("/metrics", metrics.metrics_app)


include_routers(app)
