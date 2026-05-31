from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_api_key:str
    model:str="gemini-3.5-flash"
    max_tokens:int=4096
    max_researchers:int=2
    researcher_timeout:float=30.0
    max_sub_questions:int=3

    class Config:
        env_file=".env"

settings=Settings()
        