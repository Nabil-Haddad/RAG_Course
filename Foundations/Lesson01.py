# LESSON 1: Environment Setup & Secrets Management

# Lesson Content:
#   1. How virtual environments work and why you need them
#   2. How to load secrets safely from a .env file
#   3. How to write a Config class that validates on startup
#   4. What belongs in .gitignore


# PART 1 — VIRTUAL ENVIRONMENTS

#   Every RAG project needs an isolated Python environment.
#   LangChain ships breaking changes regularly, without isolation,
#   upgrading one project silently destroys another.


# Command for VIRTUAL ENVIRONMENTS

# create the virtual environment => python -m venv .ragenv
# activate it on linux/mac => source .ragenv/bin/activate

# install the dotenv pakage => pip install python-dotenv
# lock all installed version => pip freeze > requirements.txt

# Every `pip install` now goes into .venv/ — not your global Python.


# PART 2 — THE .env FILE

# Create a file called `.env` in your project root.
# CRITICAL: also create `.env.example` with placeholder values — commit THAT,
# never the real .env. Your .gitignore must contain `.env` on its own line.


# PART 3 — LOADING SECRETS WITH python-dotenv

import os
from dotenv import load_dotenv

# load_dotenv() reads the .env file and inject each value/pairs into os.getenv()
# and we could access that using os.getenv()
# if .env doesn't exist nothings crash

# load .env values into os.getenv()
load_dotenv()

# os.getenv(key, default) => returns the default if key is missing
# if we don't specify the default value a keyError will be raised

# load those values in this script
openai_key = os.getenv("OPENAI_API_KEY", "")
environment = os.getenv("ENVIRONMENT" , "environment")
log_level = os.getenv("LOG_LEVEL", "INFO")


# we don't print the real values it is not secure
if openai_key:
    print(f"OPENAI_API_KEY : set (starts with '{openai_key[:8]}...')")
else:
    print(f"OPENAI_API_KEY : NOT SET — add it to your .env file")


# PART 4 — A CONFIG CLASS THAT VALIDATES AT STARTUP

# Most production projects use a centralise class so: 
# - Every module import from config.py not always calling os.getenv()
# - Catching missing keys immediately when the app starts


class Config:
    # we import this wherever we need settings

    # class level attributes to read from .env file
    OPENAI_API_KEY : str = os.getenv("OPENAI_API_KEY", "")
    COHERE_API_KEY : str = os.getenv("COHERE_API_KEY", "")
    ENVIRONMENT : str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL : str = os.getenv("LOG_LEVEL", "INFO")

    # Derived settings
    IS_PRODUCTION  : bool = os.getenv("ENVIRONMENT", "development") == "production"

    @classmethod
    def validate(cls)-> bool:
        # used to validate the pipline and make sure no info is missing
        required = ["OPENAI_API_KEY"]
        missing = [key for key in required if not getattr(cls ,key)]

        if missing:
            raise EnvironmentError(f"Missing required environment variables: {missing}\n")

    @classmethod
    def summary(cls) -> dict:
        return {
            "OPENAI_API_KEY" : "set" if cls.OPENAI_API_KEY else "MISSING",
            "COHERE_API_KEY" : "set" if cls.COHERE_API_KEY else "not set",
            "ENVIRONMENT" : cls.ENVIRONMENT,
            "LOG_LEVEL" : cls.LOG_LEVEL,
            "IS_PRODUCTION" : cls.IS_PRODUCTION,
        }

print(Config.summary())




# PART 5 — WHAT GOES IN .gitignore

# Create a file called `.gitignore` in the project root.
# Paste this content into it exactly:

#   .env
#   .venv/
#   __pycache__/
#   *.pyc
#   *.pyo
#   data/raw/
#   *.egg-info/
#   .DS_Store
#   .ipynb_checkpoints/

# The first two lines are the critical ones:
#   .env      — prevents your API keys from ever being committed
#   .venv/    — the virtual environment folder is large and not needed in git


