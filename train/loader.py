import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json

# Define the stock ticker and the time frame you're interested in
stock_ticker = 'PLTR'  # Palantir Technologies Inc.
start_date = (datetime.now() - timedelta(days=7)).strftime(
    '%Y-%m-%d')  # Assuming we want the last 7 days
end_date = datetime.now().strftime('%Y-%m-%d')

# Fetch 1 minute interval data using yfinance
data = yf.download(stock_ticker, start=start_date, end=end_date, interval='1m')


# Define the token creation function
def create_token(row):
  # Determine if the candle is green or red
  color = "<GREEN>" if row['Close'] >= row['Open'] else "<RED>"
  # Format numbers to have consistent decimal places
  open_p, high_p, low_p, close_p = [
      f"{x:.2f}" for x in (row['Open'], row['High'], row['Low'], row['Close'])
  ]
  # Create the token
  token = f"{color}<O={open_p}><H={high_p}><L={low_p}><C={close_p}>"
  return token


# Apply the function to each row in the DataFrame to create tokens
data['Token'] = data.apply(create_token, axis=1)

# Generate fine-tuning data
fine_tuning_data = []
prompt_length = 25  # How many tokens the model sees before making a prediction
completion_length = 5  # How many tokens ahead the model should predict for the completion

for i in range(len(data) - prompt_length - completion_length + 1):
  # Get a sequence of tokens for the prompt
  prompt_sequence = data['Token'].iloc[i:i + prompt_length].values
  # Get a sequence of tokens for the completion
  completion_sequence = data['Token'].iloc[i + prompt_length:i +
                                           prompt_length +
                                           completion_length].values

  # Create the fine-tuning data in the required format
  fine_tuning_example = {
      "messages": [{
          "role":
          "system",
          "content":
          "You tokenize prices of stocks and predict the next token"
      }, {
          "role": "user",
          "content": " ".join(prompt_sequence)
      }, {
          "role": "assistant",
          "content": " ".join(completion_sequence)
      }]
  }
  fine_tuning_data.append(fine_tuning_example)

# Define the split ratio for training and validation
split_ratio = 0.8  # 80% of the data will be used for training
split_index = int(len(fine_tuning_data) * split_ratio)

# Split the data into training and validation sets
training_data = fine_tuning_data[:split_index]
validation_data = fine_tuning_data[split_index:]

# Save the training data in the .jsonl format
training_file_path = 'training_data.jsonl'
with open(training_file_path, 'w') as outfile:
  for example in training_data:
    outfile.write(json.dumps(example) + '\n')

# Save the validation data in the .jsonl format
validation_file_path = 'validation_data.jsonl'
with open(validation_file_path, 'w') as outfile:
  for example in validation_data:
    outfile.write(json.dumps(example) + '\n')

print(f"Training data saved to {training_file_path}")
print(f"Validation data saved to {validation_file_path}")
