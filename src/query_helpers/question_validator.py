
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from utils.model_context import get_watsonx_predictor

from utils.logger import Logger
from utils import config
from src.llms.llm_loader import LLMLoader
from src import constants


class QuestionValidator:
    """
    This class is responsible for validating questions and providing one-word responses.
    """

    def __init__(self):
        """
        Initializes the QuestionValidator instance.
        """
        self.logger = Logger("question_validator").logger

    def validate_question(self, conversation, query, **kwargs):
        """
        Validates a question and provides a one-word response.

        Args:
        - conversation (str): The identifier for the conversation or task context.
        - query (str): The question to be validated.
        - **kwargs: Additional keyword arguments for customization.

        Returns:
        - list: A list of one-word responses.
        """

        model = config.ols_config.validator_model
        provider = config.ols_config.validator_provider
        verbose = kwargs.get("verbose", "").lower() == "true"

        settings_string = f"conversation: {conversation}, query: {query}, provider: {provider}, model: {model}, verbose: {verbose}"
        self.logger.info(f"{conversation} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            constants.QUESTION_VALIDATOR_PROMPT_TEMPLATE
        )

        self.logger.info(f"{conversation} Validating query")
        self.logger.info(f"{conversation} using model: {model}")

        bare_llm = LLMLoader(provider, model).llm

        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        task_query = prompt_instructions.format(query=query)

        self.logger.info(f"{conversation} task query: {task_query}")

        response = llm_chain(inputs={"query": query})
        clean_response = str(response["text"]).strip()

        self.logger.info(f"{conversation} response: {clean_response}")

        if response["text"] not in ["INVALID,NOYAML", "VALID,NOYAML", "VALID,YAML"]:
            raise ValueError("Returned response did not match the expected format")

        # will return an array:
        # [INVALID,NOYAML]
        # [VALID,NOYAML]
        # [VALID,YAML]
        return clean_response.split(",")
