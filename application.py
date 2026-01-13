from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    # Always define results for safety
    if request.method == "GET":
        return render_template("home.html", results=None)

    try:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),  # ✅ FIXED
            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get(
                "test_preparation_course"
            ),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        pred_df = data.get_data_as_data_frame()

        pipeline = PredictPipeline()
        result = pipeline.predict(pred_df)

        return render_template(
            "home.html",
            results=round(float(result[0]), 2)
        )

    except Exception as e:
        print("ERROR:", e)
        return render_template("home.html", results=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
