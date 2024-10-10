import re
import json
import random

from torch.utils.data import Dataset

class PileDataset(Dataset):
    def __init__(self, file):
        self.raw_data = self.parse_file(file)

    def parse_file(self, file):
        ret = []
        record = set()
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)['text']
                if data not in record:
                    record.add(data)
                    ret.append(data)
        return ret

    def __getitem__(self, index):
        data = self.raw_data[index]
        return data

    def __len__(self):
        return len(self.raw_data)
    
    def shuffle(self):
        random.shuffle(self.raw_data)

class PwCDataset(Dataset):
    def __init__(self, file):
        self.raw_data = self.parse_file(file)

    def parse_file(self, file):
        ret = []
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if self.not_english(data['input']): continue
                ret.append(data)
        return ret
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        data = self.raw_data[index]
        context = data['input']
        prompt = data['prompt']
        answer = data['answer']
        return (context, prompt, answer)
    
    def shuffle(self):
        random.shuffle(self.raw_data)

    def not_english(self, text):
        pattern = re.compile(r'[\u4E00-\u9FFF\u3040-\u30FF\uFF00-\uFFEF]+')
        match = pattern.search(text)
        return bool(match)
    
class PwCWithTemplate(PwCDataset):
    def __getitem__(self, index):
        context, prompt, answer = super().__getitem__(index)
        prompt = "\n\nPrompt: " + prompt
        return (context, prompt, answer)
    
class PwCForTest(PwCDataset):
    def parse_file(self, file):
        ret = []
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # For evaluation convinience, we only select first 10 questions
                if data['prompt'] == "Write a paragraph (i.e., continuation) that follows the above text.":
                    continue
                if data['prompt'] == "Rephrase the above text.":
                    continue
                if data['prompt'] == "Summarize the above text.":
                    continue
                if data['prompt'] == "Write a title for the above text.":
                    continue
                if data['prompt'] == "Extract a few keywords for the above text.":
                    continue
                if self.not_english(data['input']):
                    continue
                ret.append(data)
        return ret

    def __getitem__(self, index):
        context, prompt, answer = super().__getitem__(index)
        prompt = "\n\nPrompt: " + prompt
        return (context, prompt, answer)
