from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    # IMPORTANT: results is ALWAYS defined
    if request.method == "GET":
        return render_template("home.html", results=None)

    data = CustomData(
        gender=request.form["gender"],
        race_ethnicity=request.form["ethnicity"],
        parental_level_of_education=request.form["parental_level_of_education"],
        lunch=request.form["lunch"],
        test_preparation_course=request.form["test_preparation_course"],
        reading_score=float(request.form["reading_score"]),
        writing_score=float(request.form["writing_score"]),
    )

    pred_df = data.get_data_as_data_frame()

    pipeline = PredictPipeline()
    result = pipeline.predict(pred_df)

    return render_template("home.html", results=result[0])


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000, debug=True)
