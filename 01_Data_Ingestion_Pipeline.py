import yaml
import os
import json
import time
import pandas as pd
import datetime
import requests
import boto3
from botocore.exceptions import ClientError
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# ==========================================
# 1. AWS CLOUD & CREDENTIALS MANAGEMENT
# ==========================================
IS_AWS = os.environ.get('AWS_EXECUTION_ENV') is not None
ENV_KEY = 'enterprise_production' if IS_AWS else 'test_environment'

def get_credentials():
    """Dynamically routes credential fetches based on execution environment."""
    if IS_AWS:
        try:
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId='frye_api_keys')
            return json.loads(response['SecretString'])
        except Exception as e:
            print(f"🛑 AWS Secrets Manager Error: {e}")
            return {}
    else:
        load_dotenv()
        return os.environ

creds = get_credentials()

def get_unique_bucket_name():
    """Appends AWS Account ID to prevent global namespace collisions."""
    if IS_AWS:
        account_id = boto3.client('sts').get_caller_identity().get('Account')
        return f"frye-data-lake-{account_id}"
    return "./local_data_lake"

CONFIG_PATH = 'frye_ingestion_config.yaml'
lock = Lock()

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# ==========================================
# 2. RATE LIMITING & METRICS
# ==========================================
class RateLimiter:
    def __init__(self):
        self.last_called = {}
    def enforce_rps(self, api_name, rps_limit):
        with lock:
            current_time = time.time()
            if api_name in self.last_called:
                elapsed = current_time - self.last_called[api_name]
                if elapsed < (1.0 / rps_limit):
                    time.sleep((1.0 / rps_limit) - elapsed)
            self.last_called[api_name] = time.time()

limiter = RateLimiter()

# ==========================================
# 3. REAL-WORLD DATA RETRIEVAL
# ==========================================
def fetch_macro_data():
    fuel_price, ex_delta = 2.64, 1.0000
    eia_key, forex_key = creds.get('EIA_API_KEY'), creds.get('EXCHANGERATE_API_KEY')

    if eia_key:
        try:
            res = requests.get(config['macro_economic_sources']['jet_fuel_index']['base_url'], params={'api_key': eia_key, 'frequency': 'daily', 'data[0]': 'value'}, timeout=5)
            if res.status_code == 200: fuel_price = float(res.json()['response']['data'][0]['value'])
        except: pass

    if forex_key:
        try:
            res = requests.get(f"{config['macro_economic_sources']['currency_exchange']['base_url']}{forex_key}/latest/USD", timeout=5)
            if res.status_code == 200: ex_delta = float(res.json()['conversion_rates'].get('INR', 1.0))
        except: pass

    return {'fuel_price_index': fuel_price, 'exchange_rate_delta': ex_delta, 'airport_tax_fee': 62.50}

def fetch_aviation_intelligence(route):
    """Hits AviationStack ONLY ONCE per route to conserve extreme API limits."""
    av_conf = config['aviation_intelligence']
    limiter.enforce_rps('aviationstack_api', av_conf['rps_limits'][ENV_KEY])
    comp_count, route_pop = 2, 75.0
    api_key = creds.get('AVIATIONSTACK_API_KEY')

    if api_key:
        try:
            res = requests.get(av_conf['base_url'], params={'access_key': api_key, 'dep_iata': route.split('-')[0], 'arr_iata': route.split('-')[1]}, timeout=10)
            if res.status_code == 200:
                flights = res.json().get('data', [])
                unique_carriers = set(f['airline']['iata'] for f in flights if f.get('airline', {}).get('iata'))
                comp_count = len(unique_carriers) if len(unique_carriers) > 0 else 1
                route_pop = min(100.0, len(flights) * 2.0)
        except: pass
    return {'route_popularity': route_pop, 'carrier_competition_count': comp_count, 'airport_hub_type': 1}

def fetch_gds_data(route, date_target):
    api_conf = config['gds_pricing_api']['profiles']['self_service']
    limiter.enforce_rps('gds_api', api_conf['rps_limits'][ENV_KEY])

    try:
        auth_res = requests.post(api_conf['auth_url'], data={'grant_type': 'client_credentials', 'client_id': creds.get('AMADEUS_CLIENT_ID'), 'client_secret': creds.get('AMADEUS_CLIENT_SECRET')}, timeout=5)
        if auth_res.status_code == 200:
            token = auth_res.json().get('access_token')
            res = requests.get(api_conf['base_url'], headers={'Authorization': f'Bearer {token}'}, params={'originLocationCode': route.split('-')[0], 'destinationLocationCode': route.split('-')[1], 'departureDate': date_target.strftime("%Y-%m-%d"), 'adults': 1, 'max': 1}, timeout=10)
            if res.status_code == 200 and res.json().get('data'):
                offer = res.json()['data'][0]
                fare_details = offer['travelerPricings'][0]['fareDetailsBySegment'][0]
                return {
                    'competitor_price': float(offer['price']['total']),
                    'is_direct': 1 if len(offer['itineraries'][0]['segments']) == 1 else 0,
                    'booking_window_days': (date_target.date() - datetime.datetime.now().date()).days,
                    'service_class_tier': fare_details.get('cabin', 'ECONOMY'),
                    'bucket_tier': fare_details.get('class', 'Y')
                }
    except: pass
    return {}

# ==========================================
# 4. OPTIMIZED PIPELINE ARCHITECTURE
# ==========================================
def process_single_route(route, current_time, macro_data):
    records = []
    # Fetch physical intelligence once
    aviation_data = fetch_aviation_intelligence(route)

    # ✈️ MULTIPLY DATA YIELD: Poll Amadeus for 3 distinct lead times
    lead_times = [7, 14, 28]

    for days_out in lead_times:
        target_date = current_time + datetime.timedelta(days=days_out)
        gds_data = fetch_gds_data(route, target_date)

        if gds_data:
            records.append({
                'timestamp': current_time,
                'route_id': route,
                **gds_data,
                **aviation_data,
                'day_of_week': target_date.weekday(),
                'seasonality_index': 1.0,
                'time_of_flight_block': 3,
                **macro_data,
                'ancillary_score': 0.72
            })
    return records

def load_active_routes():
    """Loads target routes from a CSV file, falling back to defaults if missing."""
    csv_path = 'active_routes.csv'

    try:
        df_routes = pd.read_csv(csv_path)
        # Assumes the CSV has a column header named 'route_id' (e.g., JFK-LHR)
        if 'route_id' in df_routes.columns:
            routes = df_routes['route_id'].dropna().astype(str).unique().tolist()
            print(f"📍 Loaded {len(routes)} active routes from CSV.")
            return routes
        else:
            print(f"⚠️ Column 'route_id' missing in {csv_path}. Falling back to default routes.")
    except FileNotFoundError:
        print(f"⚠️ {csv_path} not found. Falling back to default routes.")
    except Exception as e:
        print(f"⚠️ Error reading routes: {e}. Falling back to defaults.")

    return ["JFK-LHR", "BLR-DEL", "DXB-BOM"]

def execute_pipeline():
    target_routes = load_active_routes()
    master_records = []
    current_time = datetime.datetime.now()
    macro_data = fetch_macro_data()

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(process_single_route, r, current_time, macro_data): r for r in target_routes}
        for future in as_completed(future_map):
            result = future.result()
            master_records.extend(result)

    df = pd.DataFrame(master_records)

    if df.empty:
        print("❌ Data yield zero. Halting upload.")
        return

    file_name = f"frye_lake_{current_time.strftime('%Y%m%d_%H%M')}.parquet"
    bucket = get_unique_bucket_name()

    if IS_AWS:
        # Write to Lambda ephemeral storage, then push to S3
        tmp_path = f"/tmp/{file_name}"
        df.to_parquet(tmp_path, engine='pyarrow', index=False)
        s3 = boto3.client('s3')
        s3_key = f"raw_data/{datetime.datetime.now().strftime('%Y-%m')}/{file_name}"

        # PROACTIVE BUCKET CHECK & AUTO-CREATE
        try:
            # Check if the bucket exists (Throws a 404 ClientError if missing)
            s3.head_bucket(Bucket=bucket)
        except ClientError as e:
            # A 404 error explicitly means "NoSuchBucket"
            if e.response['Error']['Code'] == '404':
                print(f"⚠️ Bucket '{bucket}' not found. Auto-creating now...")
                region = os.environ.get('AWS_REGION', 'us-east-1')

                # AWS mathematically requires a LocationConstraint for any region except us-east-1
                if region == 'us-east-1':
                    s3.create_bucket(Bucket=bucket)
                else:
                    s3.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                print(f"✅ Bucket '{bucket}' created successfully.")
            else:
                # Re-raise if it is an IAM permission error (e.g., 403 Forbidden)
                raise e

        # Now safely upload the file
        s3.upload_file(tmp_path, bucket, s3_key)
        print(f"☁️ Streamed optimized matrix to s3://{bucket}/{s3_key}")
    else:
        os.makedirs(bucket, exist_ok=True)
        df.to_parquet(os.path.join(bucket, file_name), engine='pyarrow', index=False)
        print(f"💻 Committed locally.")

# ==========================================
# 5. AWS EVENTBRIDGE AUTO-KILL SWITCH
# ==========================================
def manage_lifecycle():
    """Monitors ingestion lifecycle. Kills EventBridge cron if > 15 days."""
    start_date_str = os.environ.get('INGESTION_START_DATE')
    rule_name = os.environ.get('EVENTBRIDGE_RULE_NAME', 'frye-ingestion-cron')

    if start_date_str and IS_AWS:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
        if (datetime.datetime.now() - start_date).days >= 15:
            try:
                boto3.client('events').disable_rule(Name=rule_name)
                print(f"🛑 LIFECYCLE COMPLETE: Disabled EventBridge Rule '{rule_name}'.")
            except Exception as e:
                print(f"⚠️ Failed to disable EventBridge rule: {e}")

# AWS Lambda Entry Point
def lambda_handler(event, context):
    print("🚀 Firing FRYE Cloud Data Harvester...")
    manage_lifecycle()
    execute_pipeline()
    return {"statusCode": 200, "body": "Pipeline Executed Successfully"}

# Local Test Entry Point
if __name__ == "__main__":
    execute_pipeline()
