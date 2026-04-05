import json
import os
import urllib.request


URL = os.getenv("AML_ENDPOINT_URL", "https://mfailure-endpt-be578af5.ukwest.inference.ml.azure.com/score")
API_KEY = os.getenv(
  "AML_ENDPOINT_KEY",
  "3MoMA52mymFUlg9AwWDOO9616PQiYYP1a6hPLB1t5g96SHe6btEvJQQJ99CDAAAAAAAAAAAAINFRAZML3HD6",
)

if not API_KEY:
  raise Exception("A key should be provided to invoke the endpoint")


def invoke_endpoint(payload_name, payload):
  body = str.encode(json.dumps(payload))
  headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "Bearer " + API_KEY,
  }
  req = urllib.request.Request(URL, body, headers)

  try:
    response = urllib.request.urlopen(req)
    result = json.loads(response.read().decode("utf-8"))

    pred = result.get("predictions", [None])[0]
    probs = result.get("probabilities", [[None, None]])[0]
    p_failure = probs[1] if len(probs) > 1 else None

    print(f"\n=== {payload_name} ===")
    print(f"Prediction: {pred}")
    print(f"Probability (failure=1): {p_failure}")
    print(f"Raw response: {result}")
    return pred, p_failure, result
  except urllib.error.HTTPError as error:
    print(f"\n=== {payload_name} FAILED ===")
    print("The request failed with status code: " + str(error.code))
    print(error.info())
    print(error.read().decode("utf8", "ignore"))
    return None, None, None
  except urllib.error.URLError as error:
    print(f"\n=== {payload_name} FAILED ===")
    print("Endpoint connection failed. This usually means deployment is not healthy.")
    print(f"Reason: {error.reason}")
    return None, None, None


healthy_payload = {
  "data": [
    {
      "Air temperature [K]": 298.1,
      "Process temperature [K]": 308.6,
      "Rotational speed [rpm]": 1551,
      "Torque [Nm]": 42.8,
      "Tool wear [min]": 10,
    }
  ]
}

critical_failure_payload = {
  "data": [
    {
      "Air temperature [K]": 298.0,
      "Process temperature [K]": 900.0,
      "Rotational speed [rpm]": 15000,
      "Torque [Nm]": 70.0,
      "Tool wear [min]": 240,
    }
  ]
}


print(f"Endpoint: {URL}")
healthy_pred, healthy_prob, _ = invoke_endpoint("Healthy Machine Payload", healthy_payload)
critical_pred, critical_prob, _ = invoke_endpoint("Critical Failure Payload", critical_failure_payload)

if healthy_pred is not None and critical_pred is not None:
  print("\n=== Comparison Summary ===")
  print(f"Healthy prediction: {healthy_pred}, failure probability: {healthy_prob}")
  print(f"Critical prediction: {critical_pred}, failure probability: {critical_prob}")
  if healthy_pred == critical_pred and healthy_prob == critical_prob:
    print("WARNING: Outputs are identical. This usually indicates model/preprocessing mismatch or stale deployment assets.")
