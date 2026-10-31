import os
import requests

API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"


def get_mandi_prices(state=None, commodity=None, limit=100):
    """Fetch daily mandi prices from India's official government API."""

    api_key = os.getenv("DATA_GOV_IN_API_KEY")

    if not api_key:
        return {
            "success": False,
            "message": "Government API key is not configured.",
            "data": [],
        }

    params = {
        "api-key": api_key,
        "format": "json",
        "limit": min(max(int(limit), 1), 1000),
        "sort[arrival_date]": "desc",
    }

    filters = {}

    if state:
        filters["filters[state.keyword]"] = state

    if commodity:
        filters["filters[commodity]"] = commodity

    params.update(filters)

    try:
        response = requests.get(API_URL, params=params, timeout=15)
        response.raise_for_status()
        result = response.json()

        records = result.get("records", [])

        return {
            "success": True,
            "message": "Mandi prices fetched successfully.",
            "data": records,
        }

    except (requests.RequestException, ValueError) as exc:
        return {
            "success": False,
            "message": f"Unable to fetch mandi prices: {exc}",
            "data": [],
        }