from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI()

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Giải thích về {topic} cho người mới bắt đầu."
)

chain = LLMChain(prompt=prompt, llm=llm)

print(chain.run("machine learning"))
