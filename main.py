from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
Who Is Elon Musk?
Elon Musk is the founder of SpaceX and Tesla Motors, among other companies. Their success has propelled him to become the richest person in the world, with a net worth greater than $386 billion. The South African–born entrepreneur started his first businesses in the 1990s: an online city guide called Zip2 and a digital payment company named X.com, which later became PayPal. Musk, who became an American citizen in 2002, then moved into the transportation industry via his most recognizable companies. He diversified his holdings in 2022 by purchasing Twitter, which he renamed X. A vehement supporter of U.S. President Donald Trump, Musk was head of the Department of Government Efficiency, the headline-grabbing advisory group under Trump’s second administration, before announcing his departure in May 2025. Beyond his professional life, Musk is a father to 14 kids.
Where Is Elon Musk From?
Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa. His mother, Maye Musk, is a Canadian model and the oldest woman to star in a CoverGirl campaign. When Elon was growing up, she worked five jobs at one point to support her family. Elon’s father, Errol Musk, is a wealthy South African engineer.

Elon spent his early childhood with his brother, Kimbal, and sister, Tosca, in South Africa. His parents divorced when he was 10.

Around this time, Musk developed an interest in computers and taught himself how to program. When he was 12 years old, Musk developed his own game called Blastar that he sold for $500. He was often so lost in his daydreams about inventions that his parents and doctors ordered a test to check his hearing.

The short, introverted, and bookish child was bullied until he was 15. By that point, he went through a growth spurt and learned how to defend himself with karate and wrestling. He has continued to practice martial arts as an adult.

At age 17, in 1988, Elon moved to Canada against his parents’ wishes to attend university and avoid mandatory service in the South African military. He obtained his Canadian citizenship that year, in part because he felt it would be easier to obtain U.S. citizenship via that path. Elon moved to the United States a few years later and officially became an American citizen in 2002.

“I came to North America because I felt this was where there was opportunity to do great things in technology,” he said in 2013.
    """
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5", base_url="https://api.openai-proxy.org/v1")
    # llm = ChatOllama(temperature=0, model="llama3.2:latest")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
