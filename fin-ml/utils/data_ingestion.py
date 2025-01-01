import requests
import time
from requests.exceptions import HTTPError, Timeout, RequestException



def make_request(
    method,
    url,
    params=None,
    data=None,
    json=None,
    headers=None,
    max_retries=3,
    backoff_factor=1,
    timeout=10,
):
    """
    Make an HTTP request to a custom endpoint with retry logic and error handling.

    Args:
        method (str): HTTP method (e.g., 'GET', 'POST', 'PUT', 'DELETE').
        url (str): The target URL.
        params (dict): Query parameters for the request.
        data (dict): Form data for the request.
        json (dict): JSON payload for the request.
        headers (dict): Headers for the request.
        max_retries (int): Maximum number of retries on failure.
        backoff_factor (int): Multiplier for exponential backoff.
        timeout (int): Timeout for the request in seconds.

    Returns:
        dict: The JSON response or None if the request failed.
    """
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts+1):
        try:
            # Make the request
            response = requests.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()  # Raise an error for HTTP status codes 4xx/5xx
            return response  # Return the response if successful

        except HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} (Attempt {attempt}/{total_attempts})")
        except Timeout:
            print(f"Timeout occurred. Retrying... (Attempt {attempt}/{total_attempts})")
        except RequestException as req_err:
            print(f"Request error: {req_err} (Attempt {attempt}/{total_attempts})")
        except ValueError:
            print("Response is not valid JSON. Please check the endpoint.")

        # Retry logic with exponential backoff
        if attempt < total_attempts:
            wait_time = backoff_factor * (2 ** (attempt - 1))
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print("Max retries reached. Request failed.")

    return None  # Return None if all retries fail




# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(r"C:\Users\mushj\OneDrive\Desktop\WORK\Financial Analytics\.env")

    API_KEY = os.getenv('FMP_API_KEY')

    url = "https://financialmodelingprep.com/api/v3/search"
    params = {"apikey": API_KEY, "query": "AAPL"}

    response = make_request("GET", url, params=params)
    if response:
        print("Response:", response)
    else:
        print("Request failed after retries.")
