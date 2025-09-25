import time
import json
from tqdm import tqdm
from openai import OpenAI
import os

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class OpenAIGenerator:
    def __init__(self, model: str = "gpt-3.5-turbo-0125") -> None:
        self.model = model
        self.client = OpenAI()

    def preprocess_message(self, message: list):
        return message

    def generate(self, messages: list[list[dict]], *, max_completion_tokens: int = 1024) -> list[str]:
        outputs = []
        for message in tqdm(messages):
            message = self.preprocess_message(message)
            response = self.client.chat.completions.create(model=self.model, messages=message, max_completion_tokens=max_completion_tokens)
            outputs.append(response.choices[0].message.content)
        return outputs

    def generate_top_logprobs(self, messages: list[list[dict]], *, max_completion_tokens: int = 1) -> list[dict[str, float]]:
        """For each message, returns a dict from token string to token logprob for the first output token only."""
        result: list[dict[str, float]] = []
        for message in tqdm(messages):
            message = self.preprocess_message(message)
            response = self.client.chat.completions.create(model=self.model, messages=message, max_completion_tokens=max_completion_tokens, logprobs=True, top_logprobs=20)
            first_top_logprobs = response.choices[0].logprobs.content[0].top_logprobs  # pyright: ignore[reportOptionalSubscript,reportOptionalMemberAccess]
            result.append({pair.token: pair.logprob for pair in first_top_logprobs})
        return result

    def generate_batch(self, messages: list[list[dict]], *, max_completion_tokens: int = 1024, batch_id_filestore='cache-batchids.txt', logprobs=False, top_logprobs=20) -> list[str]:
        
        batch_size = 10000
        batched_messages = list(chunks(messages, batch_size))
        batch_ids = []

        for batch_index, batch in enumerate(tqdm(batched_messages)):
            
            unix_time_right_now = str(int(time.time()))
            filename = 'batch-' + unix_time_right_now + '.jsonl'

            # Create a JSONL file with batch requests
            with open(filename, 'w') as f:
                for i, message in enumerate(tqdm(batch)):
                    message = self.preprocess_message(message)
                    batch_line = {
                        "custom_id": f"request-{batch_index}-{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": message,
                            "max_completion_tokens": max_completion_tokens
                        }
                    }
                    if logprobs == True:
                        batch_line['body']['logprobs'] = True
                        batch_line['body']['top_logprobs'] = top_logprobs
                    f.write(json.dumps(batch_line) + '\n')

            # Upload the batch file
            batch_input_file = self.client.files.create(
                file=open(filename, "rb"),
                purpose="batch"
            )
            batch_input_file_id = batch_input_file.id

            # Create the batch
            batch = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"atticus job {batch_index}"}
            )

            # Store the batch ID
            batch_ids.append(batch.id)

        # store batch ids to file
        with open(batch_id_filestore, 'a') as f:
            for batch_id in batch_ids:
                f.write(str(batch_id) + '\n')
        print('batch ids stored in', batch_id_filestore)
        print('batches created and uploaded.')

        # Return all the batch IDs created
        return batch_ids

    def retrieve_generated_batches(self, save_dir='processed-batches/', batch_id_filestore='cache-batchids.txt'):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        batch_ids = []
        with open(batch_id_filestore, 'r') as f:
            batch_ids = f.read().splitlines()
        print('retrieving', len(batch_ids), 'batches...')

        statuses = {}
        for batch_id in tqdm(batch_ids):
            batch_metadata = self.client.batches.retrieve(batch_id)
            batch_status = batch_metadata.status
            if batch_status not in statuses:
                statuses[batch_status] = []
            statuses[batch_status].append(batch_id)

            if batch_metadata.output_file_id != None:
                output_file_id = batch_metadata.output_file_id
                batch_response = self.client.files.content(output_file_id)
                with open(save_dir + output_file_id + '.jsonl', 'w') as f:
                    f.write(batch_response.text)
        print('Completed batches retrieved. Statuses:', statuses)

# test
a = OpenAIGenerator()
m = [[{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}]]
a.generate_batch(m,logprobs=True)
# a.retrieve_generated_batches()
print('done.')
