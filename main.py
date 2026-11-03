import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

def main() -> None:
    print("Hello from langchain-course-02!")
    
    information = """
    Elon Reeve Musk (Pretória, 28 de junho de 1971) é um empresário e empreendedor sul-africano.
    Nascido em uma família de boas condições financeiras, Elon mudou-se para o Canadá em 1989.
    Elon investiu na Tesla em 2004 e assumiu sua direção-executiva em 2008. Cofundou a OpenAI em 2015.
    """

    summary_template = """
    given the information {information} about a person I want to create:
    1. A short summary
    2. Two interesting facts about this person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    #llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    llm = ChatOllama(model="gpt-oss", temperature=0)

    
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
