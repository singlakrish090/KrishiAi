import requests


def get_weather(city="Ludhiana"):
    """Fetch current weather and a 5-day forecast for a supported city."""

    cities = {
        "Ludhiana": (30.9010, 75.8573),
        "Amritsar": (31.6340, 74.8723),
        "Jalandhar": (31.3260, 75.5762),
        "Patiala": (30.3398, 76.3869),
        "Bathinda": (30.2110, 74.9455),
        "Chandigarh": (30.7333, 76.7794),
        "Mohali": (30.7046, 76.7179),
        "Ambala": (30.3782, 76.7767),
        "Karnal": (29.6857, 76.9905),
        "Hisar": (29.1492, 75.7217),
        "Sirsa": (29.5349, 75.0289),
        "Panipat": (29.3909, 76.9635),
    }

    if city not in cities:
        return {
            "error": "Please select a supported city."
        }

    latitude, longitude = cities[city]

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,relative_humidity_2m,"
            "apparent_temperature,is_day,precipitation,"
            "weather_code,wind_speed_10m"
        ),
        "daily": (
            "weather_code,temperature_2m_max,"
            "temperature_2m_min,precipitation_probability_max"
        ),
        "timezone": "auto",
        "forecast_days": 5,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        current = data["current"]
        daily = data["daily"]

        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with heavy hail",
        }

        forecast = []

        for i, date in enumerate(daily["time"]):
            code = daily["weather_code"][i]

            forecast.append({
                "date": date,
                "condition": weather_codes.get(code, "Variable conditions"),
                "max_temp": daily["temperature_2m_max"][i],
                "min_temp": daily["temperature_2m_min"][i],
                "rain_chance": daily["precipitation_probability_max"][i],
            })

        return {
            "city": city,
            "temperature": current["temperature_2m"],
            "feels_like": current["apparent_temperature"],
            "humidity": current["relative_humidity_2m"],
            "wind_speed": current["wind_speed_10m"],
            "precipitation": current["precipitation"],
            "condition": weather_codes.get(
                current["weather_code"], "Variable conditions"
            ),
            "forecast": forecast,
            "updated_at": current["time"],
        }

    except (requests.RequestException, KeyError, ValueError, TypeError):
        return {
            "error": "Weather data is temporarily unavailable. Please try again."
        }