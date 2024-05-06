import os
import json
from typing import Annotated, Dict
from spider_env import SpiderEnv
from autogen import ConversableAgent, UserProxyAgent, config_list_from_json
from helpers import Helpers

os.environ["AUTOGEN_USE_DOCKER"] = "False"

class GroqModule:

    def _check_termination(self, msg: Dict):
        if "tool_responses" not in msg:
            return False
        json_str = msg["tool_responses"][0]["content"]
        obj = json.loads(json_str)
        return "error" not in obj or obj["error"] is None and obj["reward"] == 1

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.helpers = Helpers()

        self.gym = SpiderEnv()
        self.observation, self.info = self.gym.reset()
        self.question = self.observation["instruction"]
        self.schema = self.info["schema"]

        self.llm_config = {
            "cache_seed": 48,
            "config_list": [
                {
                    "model": os.environ.get("OPENAI_MODEL_NAME", "llama3-70b-8192"),
                    "api_key": self.api_key,
                    "base_url": os.environ.get(
                        "OPENAI_API_BASE", "https://api.groq.com/openai/v1"
                    ),
                }
            ],
        }

        self.sql_writer = ConversableAgent(
            "sql_writer",
            llm_config=self.llm_config,
            system_message="You are good at writing SQL queries.",
            is_termination_msg=self._check_termination,
        )

        self.user_proxy = UserProxyAgent(
            "user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
           
            code_execution_config={
                "language": "SQL",
                "executive_on_receive": True,
                "error_handling": True,
            },
            system_message="You are good at evaluating if the sql code is right or not. An agent is provided message that consists SQL query. Please firstly find the SQL query and then evaluate it.",
        )

        @self.sql_writer.register_for_llm(
            description="Function for executing SQL query and returning a response"
        )
        @self.user_proxy.register_for_execution()
        def execute_sql(
            reflection: Annotated[str, "Think about what to do"],
            sql: Annotated[str, "SQL query"],
        ) -> Annotated[Dict[str, str], "Dictionary with keys 'result' and 'error'"]:

            observation, reward, _, _, info = self.gym.step(sql)
            error = observation["feedback"]["error"]
            if not error and reward == 0:
                error = "The SQL query returned an incorrect result"
            if error:
                return {
                    "error": error,
                    "wrong_result": observation["feedback"]["result"],
                    "correct_result": info["gold_result"],
                }
            else:
                return {
                    "result": observation["feedback"]["result"],
                }

    def text_to_sql(self, question: str):
        prompt_template = f"Below is the schema for a SQL database: {self.schema}\n Generate a SQL query to answer the following question: {question}."

        result = self.user_proxy.initiate_chat(self.sql_writer, message=prompt_template)
        chat = result.chat_history[len(result.chat_history) - 2]
        arg = chat["tool_calls"][0]["function"]["arguments"]
        arg_dict = json.loads(arg)

        return arg_dict["sql"]
