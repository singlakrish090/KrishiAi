import os
import uuid

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from services.weather import get_weather
from app.services.wheat_model import predict_wheat_health
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
    static_url_path="/static"
)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

PAGES = {
    "home": {
        "title": "Smart Farming Starts Here",
        "subtitle": "Your intelligent farming companion.",
        "description": "Explore crop disease detection, weather forecasts and government agriculture schemes in one place.",
        "icon": "🌾"
    },
    "dashboard": {
        "title": "Farmer Dashboard",
        "subtitle": "Your farming information, all in one place.",
        "description": "Access crop health tools, weather information and government schemes.",
        "icon": "📊"
    },
    "disease": {
        "title": "AI Crop Disease Detection",
        "subtitle": "Understand your crop's health.",
        "description": "Upload a crop image to explore our planned AI-powered disease detection tool.",
        "icon": "🌿"
    },
    "weather": {
        "title": "Live Weather",
        "subtitle": "Stay informed about local conditions.",
        "description": "Weather information will be displayed here after connecting a weather service.",
        "icon": "🌦️"
    },
    "forecast": {
        "title": "Weather Forecast",
        "subtitle": "Plan ahead with weather information.",
        "description": "Upcoming weather forecasts will be available after integration with a weather provider.",
        "icon": "📅"
    },
    "schemes": {
        "title": "Government Agriculture Schemes",
        "subtitle": "Discover support available for farmers.",
        "description": "Explore agricultural assistance and official government scheme information.",
        "icon": "🏛️"
    },
    "help": {
        "title": "Farmer Help & Support Centre",
        "subtitle": "Agricultural support, just one click away.",
        "description": "Find official agriculture helplines, crop insurance assistance, farming support and useful government resources.",
        "icon": "📞"
    },
    "about": {
        "title": "About KrishiAI",
        "subtitle": "Technology designed around farmers.",
        "description": "Our mission is to make useful agricultural information easier to access through responsible AI and digital tools.",
        "icon": "💚"
    },
    "contact": {
        "title": "Contact KrishiAI",
        "subtitle": "Help us build a better farming platform.",
        "description": "We welcome your feedback as we develop useful digital tools for farmers.",
        "icon": "📩"
    }
}


@app.route("/")
def home():
    return render_template(
        "page.html",
        page=PAGES["home"],
        active="home",
        page_key="home"
    )


# IMPORTANT: Specific dashboard route comes before the generic route.
@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        active="dashboard"
    )

@app.route("/weather")
def weather_page():
    city = request.args.get("city", "Ludhiana")
    weather = get_weather(city)

    return render_template(
        "weather.html",
        active="weather",
        weather=weather,
        cities=[
            "Ludhiana", "Amritsar", "Jalandhar", "Patiala",
            "Bathinda", "Chandigarh", "Mohali", "Ambala",
            "Karnal", "Hisar", "Sirsa", "Panipat"
        ]
    )

@app.route("/disease", methods=["GET", "POST"])
def disease_page():
    error = None
    message = None
    uploaded_image = None
    prediction = None

    if request.method == "POST":
        image = request.files.get("crop_image")

        if not image or not image.filename:
            error = "Please select a crop image first."
        else:
            filename = secure_filename(image.filename)
            extension = (
                filename.rsplit(".", 1)[-1].lower()
                if "." in filename else ""
            )

            if extension not in ALLOWED_EXTENSIONS:
                error = "Please upload a JPG, JPEG, PNG or WEBP image."
            else:
                upload_folder = os.path.abspath(
                    os.path.join(
                        app.root_path, "..", "static", "uploads"
                    )
                )
                os.makedirs(upload_folder, exist_ok=True)

                unique_filename = f"{uuid.uuid4().hex}.{extension}"
                image_path = os.path.join(
                    upload_folder, unique_filename
                )

                try:
                    image.save(image_path)

                    prediction = predict_wheat_health(image_path)
                    uploaded_image = f"uploads/{unique_filename}"
                    message = "Wheat image analysed successfully!"

                except (ValueError, OSError) as exc:
                    error = (
                        "We couldn't analyse this image. "
                        "Please try a clear wheat crop image."
                    )
                    if os.path.exists(image_path):
                        os.remove(image_path)

                except Exception:
                    app.logger.exception("Wheat AI prediction failed")
                    error = (
                        "AI analysis failed. Please try again."
                    )
                    if os.path.exists(image_path):
                        os.remove(image_path)

    return render_template(
        "disease.html",
        active="disease",
        error=error,
        message=message,
        uploaded_image=uploaded_image,
        prediction=prediction
    )


@app.errorhandler(413)
def file_too_large(error):
    return "Image is too large. Please upload an image under 8 MB.", 413

# Other pages

@app.route("/<page_key>")
def show_page(page_key):
    if page_key not in PAGES or page_key in ("home", "dashboard"):
        return render_template("404.html"), 404

    if page_key == "schemes":
        return render_template(
            "schemes.html",
            active="schemes"
        )
    
    if page_key == "help":
        return render_template(
            "help.html",
            active="help"
        )
    
    if page_key == "forecast":
        city = request.args.get("city", "Ludhiana")
        weather = get_weather(city)

        return render_template(
            "weather.html",
            active="forecast",
            weather=weather,
            cities=[
                "Ludhiana", "Amritsar", "Jalandhar", "Patiala",
                "Bathinda", "Chandigarh", "Mohali", "Ambala",
                "Karnal", "Hisar", "Sirsa", "Panipat"
            ]
        )

    return render_template(
        "page.html",
        page=PAGES[page_key],
        active=page_key,
        page_key=page_key
    )

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)