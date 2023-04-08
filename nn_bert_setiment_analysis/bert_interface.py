from transformers import pipeline

# Load the sentiment analysis pipeline
nlp = pipeline("sentiment-analysis")

# Loop to take user input and get the sentiment analysis output
while True:
    user_input = input("Enter a sentence: ")
    if user_input.lower() == "quit":
        break
    else:
        result = nlp(user_input)
        print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.2f}")  