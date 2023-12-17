# workaround to disable UserWarning
import warnings
warnings.simplefilter("ignore", UserWarning)

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

import os, inspect

from utils.logger import Logger

class LLMLoader:
    """
    Note: This class loads the LLM backend libraries if the specific LLM is loaded.
    Known caveats: Currently supports a single instance/model per backend

    llm_backends    :   a string with a supported llm backend name ('openai','ollama','tgi','watson','bam').
    params          :   (optional) array of parameters to override and pass to the llm backend

    # using the class and overriding specific parameters
    llm_backend = 'ollama'
    params = {'temperature': 0.02, 'top_p': 0.95}

    llm_config = LLMLoader(llm_backend=llm_backend, params=params)
    llm_chain = LLMChain(llm=llm_config.llm, prompt=prompt)

    """

    def __init__(self, llm_backend: str = None, params: dict = None, logger=None) -> None:
        self.logger = logger if logger is not None else Logger(
            "llm_loader").logger
        self.llm_backend = llm_backend if llm_backend else os.environ.get(
            'LLM_DEFAULT', None)
        # return empty dictionary if not defined
        self.llm_params = params if params else {}
        self.llm = None
        self._set_llm_instance()

    def _set_llm_instance(self):
        self.logger.debug(f"[{inspect.stack()[0][3]}] Loading LLM instances with default {str(self.llm_backend)}")
        # convert to string to handle None or False definitions
        match str(self.llm_backend):
            case 'openai':
                self._openai_llm_instance()
            case 'ollama':
                self._ollama_llm_instance()
            case 'tgi':
                self._tgi_llm_instance()
            case 'watson':
                self._watson_llm_instance()
            case 'bam':
                self._bam_llm_instance()
            case _:
                self.logger.error(f"ERROR: Unsupported LLM {str(self.llm_backend)}")

    def _openai_llm_instance(self):
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Creating OpenAI LLM instance")
        try:
            import openai
            from langchain.llms import OpenAI 
        except e:
            self.logger.error(f"ERROR: Missing openai libraries. Skipping loading backend LLM.")
            return
        params = {
            'base_url': os.environ.get('OPENAI_API_URL', 'https://api.openai.com/v1'),
            'api_key': os.environ.get('OPENAI_API_KEY', None),
            'model': os.environ.get('OPENAI_MODEL', None),
            'model_kwargs': {}, # TODO: add model args
            'organization': os.environ.get('OPENAI_ORGANIZATION', None),
            'timeout': os.environ.get('OPENAI_TIMEOUT', None),
            'cache': None,
            'streaming': True,            
            'temperature': 0.01,
            'max_tokens': 512,
            'top_p': 0.95,
            'frequency_penalty': 1.03,
            'verbose': False
        }
        params.update(self.llm_params) # override parameters
        self.llm=OpenAI(**params)
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] OpenAI LLM instance {self.llm}")

    def _ollama_llm_instance(self):
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Creating Ollama LLM instance")
        try:
            from langchain.llms import Ollama 
        except e:
            self.logger.error(f"ERROR: Missing ollama libraries. Skipping loading backend LLM.")
            return
        params = {
            'base_url': os.environ.get('OLLAMA_API_URL', "http://127.0.0.1:11434"),
            'model': os.environ.get('OLLAMA_MODEL', 'Mistral'),
            'cache': None,
            'temperature': 0.01,
            'top_k': 10,
            'top_p': 0.95,
            'repeat_penalty': 1.03,
            'verbose': False,
            'callback_manager': CallbackManager([StreamingStdOutCallbackHandler()])
        }
        params.update(self.llm_params)  # override parameters
        self.llm = Ollama(**params)
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Ollama LLM instance {self.llm}")

    def _tgi_llm_instance(self):
        """
        Note: TGI does not support specifying the model, it is an instance per model.
        """
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Creating Hugging Face TGI LLM instance")
        try:
            from langchain.llms import HuggingFaceTextGenInference
        except e:
            self.logger.error(f"ERROR: Missing HuggingFaceTextGenInference libraries. Skipping loading backend LLM.")
            return
        params = {
            'inference_server_url': os.environ.get('TGI_API_URL', None),
            'model_kwargs': {}, # TODO: add model args
            'max_new_tokens': 512,
            'cache': None,
            'temperature': 0.01,
            'top_k': 10,
            'top_p': 0.95,
            'repetition_penalty': 1.03,
            'streaming': True,
            'verbose': False,
            'callback_manager': CallbackManager([StreamingStdOutCallbackHandler()])
        }        
        params.update(self.llm_params)  # override parameters
        self.llm = HuggingFaceTextGenInference(**params)
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Hugging Face TGI LLM instance {self.llm}")

    def _bam_llm_instance(self):
        """BAM Research Lab"""
        self.logger.debug(f"[{inspect.stack()[0][3]}] BAM LLM instance")
        try:
            # BAM Research lab
            from genai.extensions.langchain import LangChainInterface
            from genai.credentials import Credentials
            from genai.model import Model
            from genai.schemas import GenerateParams
        except e:
            self.logger.error(f"ERROR: Missing ibm-generative-ai libraries. Skipping loading backend LLM.")
            return
        # BAM Research lab
        creds = Credentials(
            api_key=self.llm_params.get('api_key') if self.llm_params.get(
                'api_key') is not None else os.environ.get('BAM_API_KEY', None),
            api_endpoint=self.llm_params.get('api_endpoint') if self.llm_params.get(
                'api_endpoint') is not None else os.environ.get('BAM_API_URL', 'https://bam-api.res.ibm.com')
        )
        model_id = self.llm_params.get('model') if self.llm_params.get(
            'model') is not None else os.environ.get('BAM_MODEL', None)
        bam_params = {'decoding_method': "sample",
                      'max_new_tokens': 512,
                      'min_new_tokens': 1,
                      'random_seed': 42,
                      'top_k': 10,
                      'top_p': 0.95,
                      'repetition_penalty': 1.03,
                      'temperature': 0.05
                      }
        bam_params.update(self.llm_params)  # override parameters
        # remove none BAM params from dictionary
        for k in ['model','api_key','api_endpoint']:
            _ = bam_params.pop(k, None)
        params = GenerateParams(**bam_params)

        self.llm = LangChainInterface(
            model=model_id,
            params=params,
            credentials=creds
        )
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] BAM LLM instance {self.llm}")

    def _watson_llm_instance(self):
        self.logger.debug(f"[{inspect.stack()[0][3]}] Watson LLM instance")
        # WatsonX (requires WansonX libraries)
        try:
            from ibm_watson_machine_learning.foundation_models import Model
            from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
            from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
        except e:
            self.logger.error(f"ERROR: Missing ibm_watson_machine_learning libraries. Skipping loading backend LLM.")
            return
        # WatsonX uses different keys
        creds = {
            # example from https://heidloff.net/article/watsonx-langchain/
            "url": self.llm_params.get('url') if self.llm_params.get(
                'url') is not None else os.environ.get('WATSON_API_URL', None),
            "apikey": self.llm_params.get('apikey') if self.llm_params.get(
                'apikey') is not None else os.environ.get('WATSON_API_KEY', None)
        }
        # WatsonX uses different mechanism for defining parameters
        params = {
            GenParams.DECODING_METHOD: self.llm_params.get('decoding_method', 'sample'),
            GenParams.MIN_NEW_TOKENS: self.llm_params.get('min_new_tokens', 1),
            GenParams.MAX_NEW_TOKENS: self.llm_params.get('max_new_tokens',512),
            GenParams.RANDOM_SEED: self.llm_params.get('random_seed', 42),
            GenParams.TEMPERATURE: self.llm_params.get('temperature', 0.05),
            GenParams.TOP_K: self.llm_params.get('top_k',10),
            GenParams.TOP_P: self.llm_params.get('top_p',0.95),
            # https://www.ibm.com/docs/en/watsonx-as-a-service?topic=models-parameters
            GenParams.REPETITION_PENALTY: self.llm_params.get('repeatition_penallty', 1.03)
        }
        # WatsonX uses different parameter names
        llm_model = Model(model_id=self.llm_params.get('model_id', os.environ.get('WATSON_MODEL', None)),
                          credentials=creds,
                          params=params,
                          project_id=self.llm_params.get(
                              'project_id', os.environ.get('WATSON_PROJECT_ID', None))
                          )
        self.llm = WatsonxLLM(model=llm_model)
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Watson LLM instance {self.llm}")

    def status(self):
        import json
        return json.dumps(self.llm.schema_json, indent=4)

