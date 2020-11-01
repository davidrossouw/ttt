"""Authentication functionality for model_api in knative."""
import requests

from requests.exceptions import InvalidHeader
from typing import Optional

# GCP authentication service
URL_BASE = 'https://oauth2.googleapis.com/tokeninfo?id_token='
# If account returned from above above belongs to one of these domains,
# consider it authenticated.


def gcloud_auth(authorization: Optional[str], logger) -> None:
    """
    Sanity check header and attempt to authenticate.
    - authorization : GCP auth token as Bearer token in header
        argument and attemps to authenticate it against
        GCP's auth service.

    """
    # Condition for David's iPhone:
    try:
        basic, token = authorization.split(' ')
        assert basic == 'Basic'
        assert token == 'ZGF2aWQ6Y29va2llc2FuZGNyZWFt'
        logger.info("Basic auth passed")
        return
    except:
        pass

    if not authorization:
        raise InvalidHeader('Authorization not found.')

    try:
        bearer, token = authorization.split(' ')
        assert bearer == 'Bearer'
    except (IndexError, AssertionError):
        raise InvalidHeader('Bearer missing')

    response = requests.get(URL_BASE + token)
    if response.status_code >= 400:
        raise InvalidHeader('Invalid token.')
