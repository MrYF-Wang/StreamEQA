MULTICHOICE_PROMPT = """You are an advanced video question-answering AI assistant. You have been provided with a video clip and a multiple-choice question related to the video. Carefully analyze the video and choose the best answer from the four options. Respond with only one letter: A, B, C, or D.

Question: {question}

Options:
{option_a}
{option_b}
{option_c}
{option_d}

The best option is:"""


OPEN_PROMPT = """You are an advanced video question-answering AI assistant. You have been provided with a video clip and a question related to the video. Carefully analyze the video and answer the question.

Question: {question}

Answer:"""


def format_prompt(item):
    question = item["question"]
    options = item.get("options")
    if not options:
        return OPEN_PROMPT.format(question=question), None
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)} for id={item.get('id')}")
    labels = ["A", "B", "C", "D"]
    normalized = []
    for label, option in zip(labels, options):
        option = str(option)
        normalized.append(option if option.startswith(f"{label}.") else f"{label}. {option}")
    return (
        MULTICHOICE_PROMPT.format(
            question=question,
            option_a=normalized[0],
            option_b=normalized[1],
            option_c=normalized[2],
            option_d=normalized[3],
        ),
        normalized,
    )


def normalize_choice(response, options=None):
    if response is None:
        return None
    response = str(response).strip()
    for choice in ["A", "B", "C", "D"]:
        if response == choice or response.startswith(choice + ".") or response.startswith(choice + ")"):
            return choice
    if options:
        for option in options:
            if response and response in option:
                return option[0]
    for token in response.replace("\n", " ").split():
        token = token.strip(" .,:;()[]{}")
        if token in {"A", "B", "C", "D"}:
            return token
    return None

