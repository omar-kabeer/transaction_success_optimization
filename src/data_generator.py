import numpy as np
import pandas as pd
from faker import Faker
import uuid
import datetime
import random
import os
from scipy import stats

class TransactionDataGenerator:
    """
    A class to generate synthetic transaction processing data with realistic patterns.
    """
    
    def __init__(self, seed=None):
        """
        Initialize the generator with optional seed for reproducibility.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)
            
        # Initialize payment methods with their probabilities and failure rates
        self.payment_methods = {
            'credit_card': {'prob': 0.45, 'failure_rate': 0.05},
            'debit_card': {'prob': 0.30, 'failure_rate': 0.04},
            'mobile_money': {'prob': 0.15, 'failure_rate': 0.08},
            'USSD': {'prob': 0.07, 'failure_rate': 0.12},
            'bank_transfer': {'prob': 0.03, 'failure_rate': 0.03}
        }
        
        # Error codes and their descriptions
        self.error_codes = {
            'E001': 'Insufficient funds',
            'E002': 'Network timeout',
            'E003': 'Invalid account',
            'E004': 'Suspected fraud',
            'E005': 'Card expired',
            'E006': 'Payment declined',
            'E007': 'Processing error',
            'E008': 'Authentication failed',
            'E009': 'Transaction limit exceeded',
            'E010': 'System unavailable'
        }
        
        # Error code probabilities by payment method
        self.error_probabilities = {
            'credit_card': {'E001': 0.2, 'E005': 0.3, 'E006': 0.3, 'E004': 0.1, 'E009': 0.1},
            'debit_card': {'E001': 0.4, 'E002': 0.2, 'E006': 0.2, 'E003': 0.1, 'E009': 0.1},
            'mobile_money': {'E002': 0.4, 'E007': 0.3, 'E008': 0.2, 'E010': 0.1},
            'USSD': {'E002': 0.5, 'E007': 0.2, 'E010': 0.3},
            'bank_transfer': {'E003': 0.3, 'E007': 0.4, 'E010': 0.3}
        }
        
        # Create a list of merchant IDs
        self.merchant_ids = [f'MERCH{str(i).zfill(5)}' for i in range(1, 101)]
        
        # Payment processors with performance metrics
        self.processors = {
            'PROC001': {'name': 'FastPay', 'failure_rate': 0.03, 'avg_speed': 1.2, 'reliability': 0.98},
            'PROC002': {'name': 'SecureTrans', 'failure_rate': 0.02, 'avg_speed': 1.8, 'reliability': 0.99},
            'PROC003': {'name': 'QuickMoney', 'failure_rate': 0.05, 'avg_speed': 0.9, 'reliability': 0.97},
            'PROC004': {'name': 'EasyProcess', 'failure_rate': 0.04, 'avg_speed': 1.5, 'reliability': 0.96},
            'PROC005': {'name': 'TrustPayments', 'failure_rate': 0.01, 'avg_speed': 2.0, 'reliability': 0.995}
        }
        
        # Regional events that might affect transaction patterns
        self.regional_events = {
            'holiday_season': {'start_date': datetime.date(2023, 12, 15), 'end_date': datetime.date(2023, 12, 31), 
                             'volume_factor': 1.5, 'failure_factor': 1.1},
            'black_friday': {'start_date': datetime.date(2023, 11, 24), 'end_date': datetime.date(2023, 11, 26), 
                           'volume_factor': 2.0, 'failure_factor': 1.2},
            'cyber_monday': {'start_date': datetime.date(2023, 11, 27), 'end_date': datetime.date(2023, 11, 27), 
                           'volume_factor': 1.8, 'failure_factor': 1.1},
            'valentines_day': {'start_date': datetime.date(2023, 2, 14), 'end_date': datetime.date(2023, 2, 14), 
                             'volume_factor': 1.3, 'failure_factor': 1.0}
        }
        
        # User-defined events (like outages)
        self.custom_events = []

    def add_custom_event(self, name, start_date, end_date, affected_payment_methods, failure_factor):
        """
        Add a custom event that affects transaction patterns.
        
        Args:
            name (str): Name of the event
            start_date (datetime.date): Start date of the event
            end_date (datetime.date): End date of the event
            affected_payment_methods (list): List of payment methods affected
            failure_factor (float): Factor by which failure rates increase
        """
        self.custom_events.append({
            'name': name,
            'start_date': start_date,
            'end_date': end_date,
            'affected_payment_methods': affected_payment_methods,
            'failure_factor': failure_factor
        })
        
    def _generate_timestamp(self, start_date, end_date):
        """Generate a random timestamp within the given range with realistic patterns."""
        # Convert to datetime
        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
        
        # Calculate total seconds in the range
        total_seconds = int((end_datetime - start_datetime).total_seconds())
        
        # Get a random number of seconds, biased toward business hours
        rand_seconds = self._biased_time_selection(total_seconds)
        
        # Create the timestamp
        timestamp = start_datetime + datetime.timedelta(seconds=rand_seconds)
        
        return timestamp
    
    def _biased_time_selection(self, total_seconds):
        """
        Select a time biased toward business hours and weekdays.
        Returns seconds from the start date.
        """
        # First, select a random second
        rand_second = random.randint(0, total_seconds - 1)
        
        # Convert to a datetime offset
        base_date = datetime.datetime(2023, 1, 1)  # Arbitrary base date
        random_date = base_date + datetime.timedelta(seconds=rand_second)
        
        # Check if it's a weekday (0-4 are Monday to Friday)
        is_weekday = random_date.weekday() < 5
        
        # Check if it's business hours (9 AM to 6 PM)
        is_business_hours = 9 <= random_date.hour < 18
        
        # Rejection sampling - higher chance of keeping times during business hours on weekdays
        if is_weekday and is_business_hours:
            acceptance_prob = 0.8  # High probability to keep business hours on weekdays
        elif is_weekday:
            acceptance_prob = 0.4  # Medium probability for non-business hours on weekdays
        elif is_business_hours:
            acceptance_prob = 0.3  # Medium-low probability for business hours on weekends
        else:
            acceptance_prob = 0.1  # Low probability for non-business hours on weekends
            
        # If rejected, try again
        if random.random() > acceptance_prob:
            return self._biased_time_selection(total_seconds)
            
        return rand_second
    
    def _generate_payment_amount(self):
        """Generate a realistic payment amount following a log-normal distribution."""
        # Log-normal distribution gives realistic transaction amounts
        # Mean around $50 with some very large transactions
        return round(np.random.lognormal(mean=3.5, sigma=1.0), 2)
    
    def _select_payment_method(self):
        """Select a payment method based on defined probabilities."""
        methods, probabilities = zip(*[(m, data['prob']) for m, data in self.payment_methods.items()])
        return np.random.choice(methods, p=probabilities)
    
    def _determine_transaction_result(self, payment_method, timestamp):
        """Determine if a transaction succeeds or fails based on payment method and external factors."""
        base_failure_rate = self.payment_methods[payment_method]['failure_rate']
        
        # Adjust failure rate based on time of day
        hour = timestamp.hour
        if hour < 6:  # Late night
            time_factor = 1.2
        elif 6 <= hour < 9:  # Early morning
            time_factor = 1.1
        elif 9 <= hour < 17:  # Business hours
            time_factor = 0.9
        elif 17 <= hour < 22:  # Evening
            time_factor = 1.0
        else:  # Late night
            time_factor = 1.2
            
        # Adjust failure rate based on day of week
        weekday = timestamp.weekday()
        if weekday < 5:  # Weekday
            day_factor = 0.95
        else:  # Weekend
            day_factor = 1.05
            
        # Check for regional events
        event_factor = 1.0
        date = timestamp.date()
        
        for event in self.regional_events.values():
            if event['start_date'] <= date <= event['end_date']:
                event_factor = event['failure_factor']
                break
                
        # Check for custom events
        for event in self.custom_events:
            if (event['start_date'] <= date <= event['end_date'] and 
                payment_method in event['affected_payment_methods']):
                event_factor = max(event_factor, event['failure_factor'])
                
        # Calculate final failure probability
        adjusted_failure_rate = base_failure_rate * time_factor * day_factor * event_factor
        
        # Determine result
        if random.random() < adjusted_failure_rate:
            return "failure"
        else:
            return "success"
    
    def _select_error_code(self, payment_method):
        """Select an appropriate error code based on payment method."""
        error_probs = self.error_probabilities[payment_method]
        codes, probabilities = zip(*error_probs.items())
        return np.random.choice(codes, p=probabilities)
    
    def _select_processor(self, payment_method):
        """Select a payment processor based on historical performance."""
        # Different processors have different specialties
        if payment_method in ['credit_card', 'debit_card']:
            processor_weights = {
                'PROC001': 0.3,
                'PROC002': 0.4,
                'PROC003': 0.1,
                'PROC004': 0.1,
                'PROC005': 0.1
            }
        elif payment_method == 'mobile_money':
            processor_weights = {
                'PROC001': 0.1,
                'PROC002': 0.1,
                'PROC003': 0.4,
                'PROC004': 0.3,
                'PROC005': 0.1
            }
        elif payment_method == 'USSD':
            processor_weights = {
                'PROC001': 0.1,
                'PROC002': 0.0,
                'PROC003': 0.3,
                'PROC004': 0.5,
                'PROC005': 0.1
            }
        else:  # bank_transfer
            processor_weights = {
                'PROC001': 0.1,
                'PROC002': 0.3,
                'PROC003': 0.1,
                'PROC004': 0.1,
                'PROC005': 0.4
            }
            
        processors = list(processor_weights.keys())
        weights = list(processor_weights.values())
        
        return np.random.choice(processors, p=weights)
    
    def _calculate_transaction_count(self, start_date, end_date, base_count):
        """Calculate the number of transactions to generate based on date range and events."""
        # Calculate days in range
        days = (end_date - start_date).days + 1
        
        # Base transactions per day
        base_per_day = base_count / days
        
        # Adjust for events
        total_transactions = 0
        current_date = start_date
        
        while current_date <= end_date:
            daily_factor = 1.0
            
            # Check for regional events
            for event in self.regional_events.values():
                if event['start_date'] <= current_date <= event['end_date']:
                    daily_factor = max(daily_factor, event['volume_factor'])
            
            # Add daily transactions
            day_transactions = int(base_per_day * daily_factor)
            
            # Vary by day of week
            weekday = current_date.weekday()
            if weekday < 5:  # Weekday
                day_transactions = int(day_transactions * 1.2)
            else:  # Weekend
                day_transactions = int(day_transactions * 0.8)
                
            total_transactions += day_transactions
            current_date += datetime.timedelta(days=1)
            
        return total_transactions
    
    def generate_transactions(self, num_transactions=1000, start_date=None, end_date=None, output_path=None, format='csv'):
        """
        Generate synthetic transaction data.
        
        Args:
            num_transactions (int): Number of transactions to generate
            start_date (datetime.date): Start date for transactions
            end_date (datetime.date): End date for transactions
            output_path (str): Path to save the output file
            format (str): Output format ('csv' or 'parquet')
            
        Returns:
            pandas.DataFrame: Generated transaction data
        """
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime.date.today() - datetime.timedelta(days=30)
        if end_date is None:
            end_date = datetime.date.today()
            
        # Adjust transaction count based on events in the date range
        adjusted_num = self._calculate_transaction_count(start_date, end_date, num_transactions)
        
        # Initialize lists to store data
        transaction_data = []
        
        for _ in range(adjusted_num):
            # Generate transaction ID
            transaction_id = str(uuid.uuid4())
            
            # Generate timestamp
            timestamp = self._generate_timestamp(start_date, end_date)
            
            # Select merchant
            merchant_id = random.choice(self.merchant_ids)
            
            # Generate payment amount
            payment_amount = self._generate_payment_amount()
            
            # Select payment method
            payment_method = self._select_payment_method()
            
            # Determine transaction result
            result = self._determine_transaction_result(payment_method, timestamp)
            
            # Select processor
            processor_id = self._select_processor(payment_method)
# Initialize transaction record
            transaction = {
                'transaction_id': transaction_id,
                'timestamp': timestamp,
                'merchant_id': merchant_id,
                'payment_amount': payment_amount,
                'payment_method': payment_method,
                'result': result,
                'processor_id': processor_id,
                'time_of_day': timestamp.strftime('%H:%M'),
                'day_of_week': timestamp.strftime('%A'),
                'error_code': None,
                'failure_reason': None
            }
            
            # Add error details if transaction failed
            if result == 'failure':
                error_code = self._select_error_code(payment_method)
                transaction['error_code'] = error_code
                transaction['failure_reason'] = self.error_codes[error_code]
            
            # Check if transaction falls within a regional event
            for event_name, event_data in self.regional_events.items():
                if event_data['start_date'] <= timestamp.date() <= event_data['end_date']:
                    transaction['regional_event'] = event_name
                    break
            else:
                # If no regional event is found, check custom events
                for event in self.custom_events:
                    if event['start_date'] <= timestamp.date() <= event['end_date']:
                        transaction['regional_event'] = event['name']
                        break
                else:
                    transaction['regional_event'] = None
            
            transaction_data.append(transaction)
        
        # Convert to DataFrame
        df = pd.DataFrame(transaction_data)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Save to file if path is provided
        if output_path is not None:
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format.lower() == 'parquet':
                df.to_parquet(output_path, index=False)
            else:
                raise ValueError("Format must be 'csv' or 'parquet'")
        
        return df
        
    def generate_batch(self, batch_config, output_dir):
        """
        Generate multiple datasets with different configurations.
        
        Args:
            batch_config (list): List of dictionaries, each containing config for one batch
            output_dir (str): Directory to save output files
        
        Returns:
            dict: Dictionary mapping batch names to generated DataFrames
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        results = {}
        
        for config in batch_config:
            name = config.get('name', f"batch_{len(results)}")
            num_transactions = config.get('num_transactions', 1000)
            start_date = config.get('start_date', datetime.date.today() - datetime.timedelta(days=30))
            end_date = config.get('end_date', datetime.date.today())
            format = config.get('format', 'csv')
            
            # Handle custom events
            if 'custom_events' in config:
                # Clear existing custom events
                self.custom_events = []
                
                # Add new custom events
                for event in config['custom_events']:
                    self.add_custom_event(
                        name=event['name'],
                        start_date=event['start_date'],
                        end_date=event['end_date'],
                        affected_payment_methods=event['affected_payment_methods'],
                        failure_factor=event['failure_factor']
                    )
            
            # Generate file path
            file_path = os.path.join(output_dir, f"{name}.{format}")
            
            # Generate transactions
            df = self.generate_transactions(
                num_transactions=num_transactions,
                start_date=start_date,
                end_date=end_date,
                output_path=file_path,
                format=format
            )
            
            results[name] = df
            
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the generator with a seed for reproducibility
    generator = TransactionDataGenerator(seed=42)
    
    # Add a custom event (e.g., network outage)
    generator.add_custom_event(
        name="network_outage",
        start_date=datetime.date(2023, 3, 15),
        end_date=datetime.date(2023, 3, 17),
        affected_payment_methods=["mobile_money", "USSD"],
        failure_factor=3.0  # Triples the failure rate
    )
    
    # Generate a single dataset
    df = generator.generate_transactions(
        num_transactions=5000,
        start_date=datetime.date(2023, 3, 1),
        end_date=datetime.date(2023, 3, 31),
        output_path="transactions_march_2023.csv",
        format="csv"
    )
    
    print(f"Generated {len(df)} transactions")
    print(df.head())
    
    # Generate multiple datasets with different configurations
    batch_config = [
        {
            "name": "normal_week",
            "num_transactions": 3000,
            "start_date": datetime.date(2023, 4, 1),
            "end_date": datetime.date(2023, 4, 7),
            "format": "csv"
        },
        {
            "name": "holiday_week",
            "num_transactions": 5000,
            "start_date": datetime.date(2023, 12, 22),
            "end_date": datetime.date(2023, 12, 28),
            "format": "parquet",
            "custom_events": [
                {
                    "name": "system_upgrade",
                    "start_date": datetime.date(2023, 12, 24),
                    "end_date": datetime.date(2023, 12, 24),
                    "affected_payment_methods": ["credit_card", "debit_card"],
                    "failure_factor": 1.5
                }
            ]
        }
    ]
    
    batch_results = generator.generate_batch(batch_config, "output_data")
    print(f"Generated {len(batch_results)} batches of data")