MQR_PROMPT = """
You are an AI language model assistant. Your task is to generate exactly {cnt} different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
Original question: {query}
Format your response in plain text as:
{example}
"""
# Sub-query 1:
# Sub-query 2:
# Sub-query 3:
CQE_PROMPT = """
Please write a passage to answer the following user questions simultaneously.
Original_query: {original_query}
{sub_query}
Format your response in plain text as:
Passage:
"""

#  Your task is to generate a passage to retrieve relevant documents from a vector database. By generating multiple perspectives within the passage, your goal is to help overcome some of the limitations of the distance-based similarity search.

CQE_PROMPT_1 = """
Please write a passage to answer the following user questions simultaneously.
Question 1: {original_query}
Question 2: {sub_query}
Format your response in plain text as:
Passage:
"""

CQE_PROMPT_ONLY = """
Please write a passage to answer the following user questions.
Question : 
{original_query}
Format your response in plain text as:
Passage:
"""

if __name__ == "__main__":
    x = KEYWORD_EXTRACTION_PROMPT.format(
        user_input="inputs_message_old[i]",
        pre_answer="output.text",
        keyword_json_format=keyword_json_format,
        keyword_json_example=keyword_json_example
    )

