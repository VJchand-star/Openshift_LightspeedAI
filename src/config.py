import os, dotenv
from .logger import Logger

class Config:
    def __init__(self,
                 logger=None):
        # Load the dotenv configuration & set defaults
        dotenv.load_dotenv()
        self.set_defaults()

        # Parse arguments & set logger
        self.logger = logger if logger else Logger(
            logger_name="default", logfile=self.logfile).logger

    def set_defaults(self):
        """
        Set default global required parameters if none was found
        """
        # set logs file
        self.logfile = os.getenv("OLS_LOGFILE", "logs/ols.logs")

        # enable local ui?
        self.enable_ui = True if os.getenv("OLS_ENABLE_UI", True) in [True,"True"] else False

        # set default LLM model
        self.base_completion_model = os.getenv("BASE_COMPLETION_MODEL", 
                                               "ibm/granite-20b-instruct-v1")
