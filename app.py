from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data = pd.read_csv("saraswati_college_chatbot_dataset.csv")

chat_history = []

def get_response(user_query):
    questions = data["question"]

    text = [user_query.lower()] + questions.tolist()

    cv = CountVectorizer()
    vectors = cv.fit_transform(text)

    similarity = cosine_similarity(vectors)

    scores = similarity[0][1:] * 100

    temp_data = data.copy()
    temp_data["Similarity_Score"] = scores

    result = temp_data.sort_values(
        by="Similarity_Score",
        ascending=False
    )

    result = result[result["Similarity_Score"] > 28]

    if len(result) == 0:
        return "Sorry, I can't help you with this question."

    return result.head(1)["answer"].values[0]


@app.route("/", methods=["GET", "POST"])
def home():
    global chat_history

    if request.method == "POST":
        query = request.form["query"]

        answer = get_response(query)

        chat_history.append({
            "user": query,
            "bot": answer
        })

    return render_template(
        "home.html",
        chats=chat_history
    )


if __name__ == "__main__":
    app.run(debug=True)