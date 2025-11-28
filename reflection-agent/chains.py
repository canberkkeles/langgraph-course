from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import ChatBedrockConverse

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc."
            "Do not include any questions in your response. Just generate recommendations."
            "Make sure to generate recommendations based on the user input and AI generated tweet"
            "Always provide at least 1 bullet point recommendation.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts."
            " Do not ask any follow-up questions, just generate a revised version of tweets based on feedback provided by the user."
            " Generate only one tweet. ",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatBedrockConverse(model="global.anthropic.claude-sonnet-4-5-20250929-v1:0")
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm