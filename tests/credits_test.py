import os
from fotor_sdk import FotorClient
from dotenv import load_dotenv


load_dotenv(override=True)


def main() -> None:
    api_key = os.environ.get("FOTOR_OPENAPI_KEY", "")
    if not api_key:
        raise SystemExit("Set FOTOR_OPENAPI_KEY before running credits_test.py")

    client = FotorClient(
        api_key=api_key,
        endpoint=os.environ.get("FOTOR_OPENAPI_ENDPOINT", "https://api-b.fotor.com"),
    )
    credits = client.get_credits_sync()
    print("credits:", credits)


if __name__ == "__main__":
    main()
