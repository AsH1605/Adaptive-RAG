import json

# Sample dataset and model response
dataset = {
    "hotpotqa": [
        {
            "question_id": "5a879ab05542996e4f30887e",
            "question_text": "The Oberoi family is part of a hotel company that has a head office in what city?",
            "level": "medium",
            "type": "bridge",
            "contexts": [
                {"idx": 2, "title": "The Oberoi Group", "paragraph_text": "The Oberoi Group is a hotel company with its head office in Delhi.", "is_supporting": True},
                {"idx": 6, "title": "Oberoi family", "paragraph_text": "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.", "is_supporting": True}
            ],
            "answers_objects": [{"spans": ["Delhi"]}]
        },
        {
            "question_id": "4d23fd63f5e87d4b7f8b1b36",
            "question_text": "What is the capital city of Japan?",
            "level": "easy",
            "type": "factoid",
            "contexts": [
                {"idx": 1, "title": "Japan", "paragraph_text": "Japan is an island country located in East Asia. Its capital city is Tokyo.", "is_supporting": True}
            ],
            "answers_objects": [{"spans": ["Tokyo"]}]
        }
    ]
}

# Function to handle queries and simulate Flask logic
def handle_query(question_text):
    try:
        # Retrieve the question and answer based on the question text
        for dataset_name, questions in dataset.items():
            for question_data in questions:
                if question_text.lower() in question_data["question_text"].lower():
                    answer = question_data["answers_objects"][0]["spans"][0]
                    output = {
                        "question": question_data["question_text"],
                        "answer": answer
                    }
                    return output
        return {"error": "Question not found in dataset."}

    except Exception as e:
        return {"error": str(e)}

# Main logic to accept terminal inputs and handle queries interactively
if __name__ == '__main__':
    print("Welcome to the Query System!")
    print("You can enter your queries below.\n")

    while True:
        # Accept user input for the query
        question_text = input("Enter your question: ").strip()

        # Get the answer
        result = handle_query(question_text)
        
        # Output the result
        print("\nAnswer:")
        print(json.dumps(result, indent=2))

        # Option to exit
        exit_option = input("\nDo you want to ask another question? (y/n): ").strip().lower()
        if exit_option != 'y':
            print("Exiting the query system.")
            break
