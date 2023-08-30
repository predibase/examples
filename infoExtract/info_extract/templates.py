EXTRACT_TEMPLATE = """You are tasked with a reading comprehension task on this passage.

{}

Directions:
- Answer the following question(s) with factoid answer(s), looking ONLY at the context.
- If the answer is not in the context, say UNDEFINED.
- ONLY GIVE THE RELEVANT ANSWER(s), NO PREAMBLES.

{}

A1:"""


VERIFY_TEMPLATE = """You are a teacher tasked with assessing a student's reading comprehension skills from a passage. You're provided with the PASSAGE, QUESTION, and ANSWER.

PASSAGE:
{}

QUESTION: {}
ANSWER: {}

If the student's answer is completely urelated to the passage, say FALSE and briefly explain why. Otherwise (if the general meaning is correct), only say TRUE."""


MULTIVERIFY_TEMPLATE = """You are a teacher tasked with assessing a student's reading comprehension skills from a passage. You're provided with the PASSAGE, and QUESTION-ANSWER pairs.

PASSAGE:
{}

QUESTION-ANSWER pairs
{}

If the student's answer is completely urelated to the passage, say FALSE and briefly explain why. Otherwise, only say TRUE.

A1 ASSESSMENT:"""


AUTOEXTRACT_TEMPLATE = """TASK: You're writing up questions for a reading comprehension test based on the following passage.

-- start of the passage --
{}
-- end of the passage --

DIRECTIONS:
The topics of the questions include but are not limited to:
- People.
- Numbers.
- Percentage, including ”%“.
- Monetary values, including unit.
- Measurements, as of weight or distance.
- Nationalities or religious or political groups.
- Buildings, airports, highways, bridges, etc.
- Companies, agencies, institutions, etc.
- Countries, cities, states.
- Languages.
- Absolute or relative dates or periods.
- Key Performance Indicators (KPIs)

Generate all question-answer pairs that are relevant to the passage. Answers must be short and factoid. Format the question-answer pairs as follows:

Question: <question>
Answer: <answer>

"""

SYNTHESIZE_TEMPLATE = """You are sitting for an exam and are presented with options that may or may not contain incorrect answers.

{}

Directions:
- Answer the following question from the options above by combining answers.
- Nicely and cleanly format the text.
- ONLY GIVE THE RELEVANT ANSWER(s), NO PREAMBLES.

Question: {}
Answer:"""


FINAL_SYNTHESIZE_TEMPLATE = """You are presented with a set of correct answer(s) for a question.

{}

Summarize the answer(s) to the following question making sure to keep ALL details.

Question: {}
Answer:"""


SQL_TEMPLATE = """
Below are examples of correct responses to the prompt, followed by a question. Use the examples to inform your final response.

EXAMPLE: Below is the data schema from a PostgreSQL table named `df`.

['age', 'gender', 'name']

English to SQL: How many rows are there in the table?

ASSISTANT: SELECT * FROM df;

EXAMPLE: Below is the data schema from a PostgreSQL table named `df`.

['species', 'population', 'endangered']

English to SQL: Can you show me the species that are endangered?

ASSISTANT: SELECT DISTINCT species FROM df WHERE endangered = true;

QUESTION: Below is the data schema from a PostgreSQL table named `df`.

{}

English to SQL: {}

ASSISTANT:"""
